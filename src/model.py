"""The GPT language model, a model factory, and checkpoint save/load helpers."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config
from src.bigram import BigramLanguageModel
from src.sampling import GenerationMixin
from src.tokenizer import CharTokenizer
from src.transformer import Block


class GPTLanguageModel(GenerationMixin, nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_embd: int, n_head: int,
                 n_layer: int, dropout: float, attn_impl: str = "fused"):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout, attn_impl) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")
        tok_emb = self.token_embedding_table(idx)                                   # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = self.blocks(tok_emb + pos_emb)
        logits = self.lm_head(self.ln_f(x))                                         # (B, T, V)

        loss = None
        if targets is not None:
            V = logits.size(-1)
            loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))
        return logits, loss


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
def build_model(cfg: Config, vocab_size: int) -> nn.Module:
    if cfg.model_type == "bigram":
        return BigramLanguageModel(vocab_size, cfg.block_size)
    return GPTLanguageModel(
        vocab_size=vocab_size, block_size=cfg.block_size, n_embd=cfg.n_embd, n_head=cfg.n_head,
        n_layer=cfg.n_layer, dropout=cfg.dropout, attn_impl=cfg.attn_impl,
    )


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------
def save_checkpoint(path: str | Path, model: nn.Module, cfg: Config, tokenizer: CharTokenizer,
                    iter_num: int, best_val_loss: float, optimizer: torch.optim.Optimizer | None = None,
                    extra: dict | None = None) -> None:
    """Save weights + config + vocabulary so a checkpoint is self-contained."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": cfg.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "iter_num": iter_num,
        "best_val_loss": float(best_val_loss),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: never leave a half-written checkpoint


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict:
    """Load a checkpoint and rebuild the model.

    Returns a dict with keys: model, tokenizer, config, iter_num, best_val_loss, extra,
    and optimizer_state (if the checkpoint has one).
    """
    ckpt = torch.load(Path(path), map_location=device)
    cfg = Config.from_dict(ckpt["config"])
    tokenizer = CharTokenizer.from_dict(ckpt["tokenizer"])
    model = build_model(cfg, tokenizer.vocab_size)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    out = {
        "model": model,
        "tokenizer": tokenizer,
        "config": cfg,
        "iter_num": ckpt["iter_num"],
        "best_val_loss": ckpt["best_val_loss"],
        "extra": ckpt.get("extra", {}),
    }
    if "optimizer_state" in ckpt:
        out["optimizer_state"] = ckpt["optimizer_state"]
    return out
