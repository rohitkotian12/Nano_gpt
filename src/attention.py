"""Causal self-attention, in two interchangeable implementations.

* `Head` / `MultiHeadAttention` - the lecture's version: one `Head` module per head,
  explicit (T, T) score matrix and a lower-triangular mask. Easy to read.
* `CausalSelfAttention` - all heads in one batched matmul using PyTorch's fused
  `scaled_dot_product_attention`. Mathematically the same, much faster.

`make_attention` picks one by name; `copy_loop_weights_to_fused` converts weights so the
two can be checked against each other (see tests/test_models.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5      # (B, T, T) affinities
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # no peeking at the future
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)                                # (B, T, hs)


class MultiHeadAttention(nn.Module):
    """Several `Head`s in parallel, concatenated and projected back to n_embd."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class CausalSelfAttention(nn.Module):
    """Batched multi-head causal attention using a single QKV projection."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.dropout_p = dropout
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        hs = C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, hs).transpose(1, 2) for t in (q, k, v))  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


def make_attention(impl: str, n_embd: int, n_head: int, block_size: int, dropout: float) -> nn.Module:
    if impl == "fused":
        return CausalSelfAttention(n_embd, n_head, block_size, dropout)
    if impl == "loop":
        return MultiHeadAttention(n_embd, n_head, block_size, dropout)
    raise ValueError(f"Unknown attention implementation {impl!r}")


@torch.no_grad()
def copy_loop_weights_to_fused(loop: MultiHeadAttention, fused: CausalSelfAttention) -> None:
    """Load a `MultiHeadAttention`'s weights into an equivalent `CausalSelfAttention`."""
    C = fused.proj.in_features
    hs = C // fused.n_head
    for h, head in enumerate(loop.heads):
        rows = slice(h * hs, (h + 1) * hs)
        fused.qkv.weight[0 * C:1 * C][rows] = head.query.weight
        fused.qkv.weight[1 * C:2 * C][rows] = head.key.weight
        fused.qkv.weight[2 * C:3 * C][rows] = head.value.weight
    fused.proj.weight.copy_(loop.proj.weight)
    fused.proj.bias.copy_(loop.proj.bias)
