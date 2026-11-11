"""Transformer building blocks: feed-forward network and the pre-norm residual block."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.attention import make_attention


class FeedForward(nn.Module):
    """Position-wise MLP: Linear -> ReLU -> Linear -> Dropout (4x expansion)."""

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication (attention) followed by computation (MLP)."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, attn_impl: str = "fused"):
        super().__init__()
        self.sa = make_attention(attn_impl, n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))   # residual connection around attention
        x = x + self.ffwd(self.ln2(x))  # residual connection around the MLP
        return x
