"""Small, equal-budget ablations on the GPT model, logged to experiments/results.csv.

Every GPT run starts from the same small base model and changes one thing at a time.
Runs are short on purpose (they must finish on a laptop CPU), so treat the numbers as
*relative* comparisons, not as final quality. Finished experiments are skipped on re-run.

Usage:
    python -m experiments.run_experiments                  # run everything not yet in results.csv
    python -m experiments.run_experiments --only lr_1e-2 depth_4
    python -m experiments.run_experiments --force          # redo all
    python -m experiments.run_experiments --report         # print results as a Markdown table
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from configs.config import PRESETS, Config, get_config
from src.train import train

RESULTS_CSV = Path(__file__).resolve().parent / "results.csv"
COLUMNS = ["experiment", "group", "model_type", "n_layer", "n_head", "n_embd", "block_size", "dropout",
           "learning_rate", "lr_schedule", "attn_impl", "batch_size", "max_iters", "params",
           "train_loss", "val_loss", "val_ppl", "val_bpc", "best_val_loss", "seconds", "seed"]

# Small base model shared by all GPT experiments.
BASE = dict(model_type="gpt", n_layer=2, n_head=4, n_embd=64, block_size=64, batch_size=32, dropout=0.0,
            learning_rate=3e-3, min_lr=3e-4, warmup_iters=20, lr_schedule="cosine", max_iters=400,
            eval_interval=200, eval_iters=50, save_checkpoints=False, save_logs=False, seed=1337)

# (name, group, overrides)
EXPERIMENTS: list[tuple[str, str, dict]] = [
    ("bigram", "baseline", {**PRESETS["bigram"], "max_iters": 3000, "eval_iters": 50,
                            "save_checkpoints": False, "save_logs": False}),
    ("gpt_base", "baseline", {}),
    ("depth_1", "depth", {"n_layer": 1}),
    ("depth_4", "depth", {"n_layer": 4}),
    ("width_32", "width", {"n_embd": 32}),
    ("width_128", "width", {"n_embd": 128}),
    ("heads_1", "heads", {"n_head": 1}),
    ("heads_2", "heads", {"n_head": 2}),
    ("heads_8", "heads", {"n_head": 8}),
    ("context_8", "context", {"block_size": 8}),
    ("context_16", "context", {"block_size": 16}),
    ("context_32", "context", {"block_size": 32}),
    ("lr_3e-4", "learning_rate", {"learning_rate": 3e-4, "min_lr": 3e-5}),
    ("lr_1e-3", "learning_rate", {"learning_rate": 1e-3, "min_lr": 1e-4}),
    ("lr_1e-2", "learning_rate", {"learning_rate": 1e-2, "min_lr": 1e-3}),
    ("dropout_0.2", "regularisation", {"dropout": 0.2}),
    ("lr_constant", "schedule", {"lr_schedule": "constant"}),
    ("attn_loop", "implementation", {"attn_impl": "loop"}),
    # Same config as gpt_base with other seeds: shows how much run-to-run noise to expect.
    ("seed_1", "noise", {"seed": 1}),
    ("seed_2", "noise", {"seed": 2}),
]


def make_config(name: str, overrides: dict, max_iters: int | None = None) -> Config:
    merged = {**BASE, **overrides, "run_name": name}
    if max_iters is not None and merged["model_type"] == "gpt":
        merged["max_iters"] = max_iters
    return Config(**merged)


def _row(name: str, group: str, cfg: Config, s: dict) -> dict:
    return {
        "experiment": name, "group": group, "model_type": cfg.model_type,
        "n_layer": cfg.n_layer if cfg.model_type == "gpt" else 0,
        "n_head": cfg.n_head if cfg.model_type == "gpt" else 0,
        "n_embd": cfg.n_embd if cfg.model_type == "gpt" else 0,
        "block_size": cfg.block_size, "dropout": cfg.dropout, "learning_rate": cfg.learning_rate,
        "lr_schedule": cfg.lr_schedule, "attn_impl": cfg.attn_impl if cfg.model_type == "gpt" else "-",
        "batch_size": cfg.batch_size, "max_iters": cfg.max_iters, "params": s["params"],
        "train_loss": round(s["train_loss"], 4), "val_loss": round(s["val_loss"], 4),
        "val_ppl": round(math.exp(s["val_loss"]), 3), "val_bpc": round(s["val_loss"] / math.log(2), 4),
        "best_val_loss": round(s["best_val_loss"], 4), "seconds": round(s["seconds"], 1), "seed": cfg.seed,
    }


def read_results(path: Path = RESULTS_CSV) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def report(path: Path = RESULTS_CSV) -> str:
    rows = read_results(path)
    head = "| experiment | group | layers | heads | embd | ctx | dropout | lr | params | val loss | val ppl | time (s) |\n"
    head += "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    body = "".join(
        f"| {r['experiment']} | {r['group']} | {r['n_layer']} | {r['n_head']} | {r['n_embd']} | {r['block_size']} | "
        f"{r['dropout']} | {r['learning_rate']} | {int(r['params']):,} | {r['val_loss']} | {r['val_ppl']} | {r['seconds']} |\n"
        for r in rows)
    return head + body


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", help="run only these experiment names")
    p.add_argument("--max_iters", type=int, default=None, help="override the GPT iteration budget")
    p.add_argument("--force", action="store_true", help="re-run experiments already in results.csv")
    p.add_argument("--report", action="store_true", help="print the results table and exit")
    p.add_argument("--list", action="store_true", help="list experiment names and exit")
    args = p.parse_args(argv)

    if args.report:
        print(report())
        return
    if args.list:
        for name, group, ov in EXPERIMENTS:
            print(f"{name:14s} [{group}] {ov if group != 'baseline' or name == 'gpt_base' else '(bigram preset)'}")
        return

    known = {name for name, _, _ in EXPERIMENTS}
    if args.only and (bad := set(args.only) - known):
        raise SystemExit(f"Unknown experiments: {sorted(bad)}. Use --list.")

    done = {r["experiment"] for r in read_results()}
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if args.force and not args.only:
        RESULTS_CSV.unlink(missing_ok=True)
        done = set()
    elif args.force:
        keep = [r for r in read_results() if r["experiment"] not in set(args.only)]
        with open(RESULTS_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(keep)
        done = {r["experiment"] for r in keep}

    for name, group, overrides in EXPERIMENTS:
        if args.only and name not in args.only:
            continue
        if name in done:
            print(f"skip {name} (already in results.csv)")
            continue
        cfg = make_config(name, overrides, args.max_iters)
        print(f"\n=== {name} [{group}] ===", flush=True)
        summary = train(cfg, verbose=False)
        row = _row(name, group, cfg, summary)
        new_file = not RESULTS_CSV.exists()
        with open(RESULTS_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            if new_file:
                w.writeheader()
            w.writerow(row)
        print(f"{name}: params={row['params']:,} val_loss={row['val_loss']} "
              f"ppl={row['val_ppl']} time={row['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
