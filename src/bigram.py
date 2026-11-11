"""Bigram baseline: each token directly looks up the logits of the next token."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sampling import GenerationMixin


class BigramLanguageModel(GenerationMixin, nn.Module):
    def __init__(self, vocab_size: int, block_size: int = 8):
        super().__init__()
        self.block_size = block_size
        # (V, V) table: row i holds the logits for the token that follows token i.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # idx, targets: (B, T) integer tensors
        logits = self.token_embedding_table(idx)  # (B, T, V)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))
        return logits, loss
