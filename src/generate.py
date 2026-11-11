"""Text generation from a trained checkpoint.

Usage:
    python -m src.generate --checkpoint checkpoints/gpt_best.pt --prompt "ROMEO:" \
        --max_new_tokens 500 --temperature 0.8 --top_k 40 --top_p 0.95 --seed 1
"""
from __future__ import annotations

import argparse
import sys
from typing import Iterator

import torch

from src.model import load_checkpoint
from src.tokenizer import CharTokenizer
from src.utils import count_parameters, format_params, get_device


def _start_context(model, tokenizer: CharTokenizer, prompt: str, device) -> tuple[torch.Tensor, str]:
    """Encode the prompt (dropping unknown chars). An empty prompt starts from a newline."""
    ids = tokenizer.encode(prompt, errors="ignore")
    if not ids:
        ids = [tokenizer.stoi.get("\n", 0)]
    return torch.tensor([ids], dtype=torch.long, device=device), tokenizer.decode(ids)


def _make_generator(seed: int | None, device) -> torch.Generator | None:
    if seed is None:
        return None
    return torch.Generator(device=torch.device(device).type).manual_seed(seed)


def stream_text(model, tokenizer: CharTokenizer, prompt: str = "", max_new_tokens: int = 500,
                temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None,
                seed: int | None = None, device: str | torch.device | None = None) -> Iterator[str]:
    """Yield the text piece by piece: first the (cleaned) prompt, then one character at a time."""
    device = device or next(model.parameters()).device
    context, start_text = _start_context(model, tokenizer, prompt, device)
    yield start_text
    gen = _make_generator(seed, device)
    for nxt in model.generate_stream(context, max_new_tokens, temperature, top_k, top_p, gen):
        yield tokenizer.decode(nxt[0])


def generate_text(model, tokenizer: CharTokenizer, prompt: str = "", max_new_tokens: int = 500,
                  temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None,
                  seed: int | None = None, device: str | torch.device | None = None) -> str:
    """Return prompt + `max_new_tokens` generated characters as one string."""
    return "".join(stream_text(model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p, seed, device))


def main() -> None:
    p = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    p.add_argument("--checkpoint", default="checkpoints/gpt_best.pt")
    p.add_argument("--prompt", default="")
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8, help="<=0 means greedy")
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--output", default=None, help="also write the samples to this file")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    model, tok, cfg = ckpt["model"], ckpt["tokenizer"], ckpt["config"]

    dropped = tok.unknown_chars(args.prompt)
    if dropped:
        print(f"warning: dropping characters not in the vocabulary: {dropped}", file=sys.stderr)

    header = (
        f"# Sample generation\n"
        f"checkpoint : {args.checkpoint}\n"
        f"model      : {cfg.model_type}, {format_params(count_parameters(model))} params, "
        f"{ckpt['iter_num']} training iters, best val loss {ckpt['best_val_loss']:.4f}\n"
        f"settings   : temperature={args.temperature} top_k={args.top_k} top_p={args.top_p} "
        f"max_new_tokens={args.max_new_tokens} seed={args.seed}\n"
    )
    blocks = [header]
    for i in range(args.num_samples):
        text = generate_text(model, tok, args.prompt, args.max_new_tokens, args.temperature,
                             args.top_k, args.top_p, args.seed + i, device)
        blocks.append(f"{'=' * 70}\n--- sample {i + 1} (prompt: {args.prompt!r}, seed {args.seed + i}) ---\n{text}\n")
    output = "\n".join(blocks)
    print(output)
    if args.output:
        from pathlib import Path
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output, encoding="utf-8")
        print(f"(saved to {out})", file=sys.stderr)


if __name__ == "__main__":
    main()
