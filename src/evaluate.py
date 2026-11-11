"""Evaluation: loss estimates, perplexity and bits-per-character."""
from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext

import torch

from configs.config import Config
from src.data_loader import DataBundle, prepare_data
from src.model import load_checkpoint
from src.utils import count_parameters, format_params, get_device


@torch.no_grad()
def estimate_loss(model: torch.nn.Module, data: DataBundle, eval_iters: int, ctx=None,
                  generator: torch.Generator | None = None) -> dict[str, float]:
    """Average loss over `eval_iters` random batches for both splits (dropout off)."""
    ctx = ctx if ctx is not None else nullcontext()
    was_training = model.training
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = data.get_batch(split, generator)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    if was_training:
        model.train()
    return out


def loss_to_metrics(loss: float) -> dict[str, float]:
    """Cross-entropy (nats/char) -> perplexity and bits-per-character."""
    return {"loss": loss, "perplexity": math.exp(loss), "bpc": loss / math.log(2)}


def evaluate_checkpoint(path: str, eval_iters: int = 200, device: str = "auto", seed: int = 0,
                        data_path: str | None = None) -> dict:
    """Evaluate a saved checkpoint on the train/val splits with a fixed sampling seed."""
    dev = get_device(device)
    ckpt = load_checkpoint(path, dev)
    cfg: Config = ckpt["config"]
    if data_path:
        cfg = cfg.with_overrides(data_path=data_path)
    data = prepare_data(cfg, tokenizer=ckpt["tokenizer"], device=dev)
    gen = torch.Generator().manual_seed(seed)
    losses = estimate_loss(ckpt["model"], data, eval_iters, generator=gen)
    return {
        "checkpoint": str(path),
        "model_type": cfg.model_type,
        "params": count_parameters(ckpt["model"]),
        "trained_iters": ckpt["iter_num"],
        "train": loss_to_metrics(losses["train"]),
        "val": loss_to_metrics(losses["val"]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint (loss / perplexity / bits-per-char).")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval_iters", type=int, default=200)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_path", default=None)
    p.add_argument("--json", action="store_true", help="print raw JSON")
    args = p.parse_args()

    r = evaluate_checkpoint(args.checkpoint, args.eval_iters, args.device, args.seed, args.data_path)
    if args.json:
        print(json.dumps(r, indent=2))
        return
    print(f"{r['checkpoint']}  [{r['model_type']}, {format_params(r['params'])} params, "
          f"{r['trained_iters']} iters]")
    for split in ("train", "val"):
        m = r[split]
        print(f"  {split:5s} loss {m['loss']:.4f} | perplexity {m['perplexity']:.2f} | {m['bpc']:.3f} bits/char")


if __name__ == "__main__":
    main()
