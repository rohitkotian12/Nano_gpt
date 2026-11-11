"""Training loop with AdamW, warmup + cosine LR, gradient clipping / accumulation,
best/last checkpoints, resume, early stopping and CSV logging.

Usage:
    python -m src.train --preset bigram
    python -m src.train --preset gpt_cpu
    python -m src.train --preset gpt                 # full lecture config (GPU recommended)
    python -m src.train --preset gpt_cpu --max_iters 500 --n_layer 2 --resume
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from contextlib import nullcontext
from dataclasses import fields

import torch
from tqdm import tqdm

from configs.config import PRESETS, Config, get_config
from src.data_loader import prepare_data
from src.evaluate import estimate_loss
from src.model import build_model, load_checkpoint, save_checkpoint
from src.utils import count_parameters, format_params, get_device, set_seed

LOG_FIELDS = ["step", "train_loss", "val_loss", "lr", "elapsed_s", "tokens_seen"]


def get_lr(it: int, cfg: Config) -> float:
    """Linear warmup, then cosine decay from `learning_rate` to `min_lr` (or constant)."""
    if cfg.warmup_iters > 0 and it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    if cfg.lr_schedule == "constant":
        return cfg.learning_rate
    if it >= cfg.max_iters:
        return cfg.min_lr
    progress = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def configure_optimizer(model: torch.nn.Module, cfg: Config, device: torch.device) -> torch.optim.Optimizer:
    """AdamW with weight decay on matrices only (not on biases / LayerNorm gains)."""
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=cfg.betas,
                             fused=True if device.type == "cuda" else None)


def train(cfg: Config, resume: bool = False, verbose: bool = True) -> dict:
    """Train a model according to `cfg`. Returns a summary dict."""
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    data = prepare_data(cfg, device=device)
    model = build_model(cfg, data.vocab_size).to(device)
    optimizer = configure_optimizer(model, cfg, device)
    n_params = count_parameters(model)

    ckpt_dir = cfg.resolve(cfg.checkpoint_dir)
    best_path = ckpt_dir / f"{cfg.run_name}_best.pt"
    last_path = ckpt_dir / f"{cfg.run_name}_last.pt"
    log_path = cfg.resolve(cfg.log_dir) / f"{cfg.run_name}.csv"

    start_iter, best_val, best_iter, bad_evals = 0, float("inf"), 0, 0
    resumed = False
    if resume and last_path.exists():
        ck = load_checkpoint(last_path, device)
        if ck["tokenizer"] != data.tokenizer:
            raise ValueError("Checkpoint vocabulary does not match the current data")
        model.load_state_dict(ck["model"].state_dict())
        model.train()
        if "optimizer_state" in ck:
            optimizer.load_state_dict(ck["optimizer_state"])
        start_iter, best_val = ck["iter_num"], ck["best_val_loss"]
        best_iter = ck["extra"].get("best_iter", start_iter)
        bad_evals = ck["extra"].get("bad_evals", 0)
        resumed = True
        if verbose:
            print(f"Resumed from {last_path} at iter {start_iter}")

    if cfg.save_logs:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not (resumed and log_path.exists())
        log_file = open(log_path, "a" if resumed else "w", newline="")
        log_writer = csv.writer(log_file)
        if write_header:
            log_writer.writerow(LOG_FIELDS)
    else:
        log_file = log_writer = None

    use_amp = cfg.amp and device.type == "cuda"
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    if verbose:
        print(f"[{cfg.run_name}] {cfg.model_type} | {format_params(n_params)} params | device={device} | "
              f"vocab={data.vocab_size} | train/val tokens={len(data.train_data):,}/{len(data.val_data):,}")

    tokens_per_iter = cfg.batch_size * cfg.block_size * cfg.grad_accum_steps
    t0 = time.time()
    final_losses: dict[str, float] = {}
    stopped_early = False
    model.train()
    pbar = tqdm(range(start_iter, cfg.max_iters + 1), disable=not verbose, desc="train", leave=False)

    try:
        for it in pbar:
            lr = get_lr(it, cfg)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # ---- periodic evaluation (after `it` optimiser updates) -----------
            is_eval_step = it % cfg.eval_interval == 0 or it == cfg.max_iters
            if is_eval_step and not (resumed and it == start_iter):
                losses = estimate_loss(model, data, cfg.eval_iters, ctx)
                final_losses = losses
                elapsed = time.time() - t0
                if log_writer:
                    log_writer.writerow([it, f"{losses['train']:.5f}", f"{losses['val']:.5f}", f"{lr:.6g}",
                                         f"{elapsed:.1f}", it * tokens_per_iter])
                    log_file.flush()
                if verbose:
                    tqdm.write(f"step {it:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
                               f"lr {lr:.2e}, {elapsed:.0f}s")

                if losses["val"] < best_val:
                    best_val, best_iter, bad_evals = losses["val"], it, 0
                    if cfg.save_checkpoints:
                        save_checkpoint(best_path, model, cfg, data.tokenizer, it, best_val)
                else:
                    bad_evals += 1
                if cfg.save_checkpoints:
                    save_checkpoint(last_path, model, cfg, data.tokenizer, it, best_val, optimizer,
                                    extra={"best_iter": best_iter, "bad_evals": bad_evals})
                if cfg.patience and bad_evals >= cfg.patience:
                    stopped_early = True
                    if verbose:
                        tqdm.write(f"Early stopping: no val improvement for {bad_evals} evaluations")
                    break

            if it == cfg.max_iters:
                break

            # ---- one optimisation step (with gradient accumulation) ------------
            optimizer.zero_grad(set_to_none=True)
            for _ in range(cfg.grad_accum_steps):
                xb, yb = data.get_batch("train")
                with ctx:
                    _, loss = model(xb, yb)
                (loss / cfg.grad_accum_steps).backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            if verbose and it % 20 == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr:.1e}")
    finally:
        pbar.close()
        if log_file:
            log_file.close()

    elapsed = time.time() - t0
    return {
        "run_name": cfg.run_name,
        "model_type": cfg.model_type,
        "params": n_params,
        "device": str(device),
        "train_loss": final_losses.get("train", float("nan")),
        "val_loss": final_losses.get("val", float("nan")),
        "best_val_loss": best_val,
        "best_iter": best_iter,
        "seconds": elapsed,
        "stopped_early": stopped_early,
        "best_checkpoint": str(best_path) if cfg.save_checkpoints else None,
        "last_checkpoint": str(last_path) if cfg.save_checkpoints else None,
        "log_path": str(log_path) if cfg.save_logs else None,
    }


# --------------------------------------------------------------------------- CLI
_TYPES = {"int": int, "float": float, "str": str, "bool": bool}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the bigram or GPT model.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--preset", default="gpt_cpu", choices=sorted(PRESETS))
    p.add_argument("--resume", action="store_true", help="continue from <run_name>_last.pt")
    p.add_argument("--quiet", action="store_true")
    for f in fields(Config):
        if f.name == "betas":
            continue
        typ = _TYPES[f.type] if isinstance(f.type, str) else f.type
        if typ is bool:
            p.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=None)
        else:
            p.add_argument(f"--{f.name}", type=typ, default=None)
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    overrides = {f.name: getattr(args, f.name) for f in fields(Config)
                 if f.name != "betas" and getattr(args, f.name) is not None}
    cfg = get_config(args.preset, **overrides)
    summary = train(cfg, resume=args.resume, verbose=not args.quiet)
    if not args.quiet:
        print(f"\nDone in {summary['seconds']:.0f}s | final val loss {summary['val_loss']:.4f} | "
              f"best val loss {summary['best_val_loss']:.4f} (iter {summary['best_iter']})")
        if summary["best_checkpoint"]:
            print(f"Best checkpoint: {summary['best_checkpoint']}")
    return summary


if __name__ == "__main__":
    main()
