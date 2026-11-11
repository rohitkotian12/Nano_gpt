"""Token sampling (temperature / top-k / top-p) and a shared generation mixin."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_logits(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                  top_p: float | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample one token id per row of `logits` (B, V) -> (B, 1).

    temperature <= 0 means greedy decoding. top_k keeps the k most likely tokens;
    top_p (nucleus) keeps the smallest set whose probability mass reaches p.
    """
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth_best = torch.topk(logits, k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth_best, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        # Drop a token if the mass *before* it already reaches top_p (so the top token always stays).
        drop_sorted = (torch.cumsum(sorted_probs, dim=-1) - sorted_probs) >= top_p
        drop = torch.zeros_like(drop_sorted).scatter(-1, sorted_idx, drop_sorted)
        logits = logits.masked_fill(drop, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


class GenerationMixin:
    """Adds `generate` / `generate_stream` to any model with `forward(idx) -> (logits, loss)`
    and a `block_size` attribute."""

    block_size: int

    @torch.no_grad()
    def generate_stream(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                        top_k: int | None = None, top_p: float | None = None,
                        generator: torch.Generator | None = None):
        """Yield one (B, 1) tensor of new token ids at a time."""
        was_training = self.training
        self.eval()  # no dropout while sampling
        try:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size:]  # crop context to the last block_size tokens
                logits, _ = self(idx_cond)
                nxt = sample_logits(logits[:, -1, :], temperature, top_k, top_p, generator)
                idx = torch.cat((idx, nxt), dim=1)
                yield nxt
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """Extend `idx` (B, T) by `max_new_tokens` sampled tokens -> (B, T + max_new_tokens)."""
        new = list(self.generate_stream(idx, max_new_tokens, temperature, top_k, top_p, generator))
        return torch.cat([idx, *new], dim=1) if new else idx
