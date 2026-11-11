"""Data loading: read text, tokenize, split 90/10 and serve random batches."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from configs.config import Config
from src.tokenizer import CharTokenizer
from src.utils import get_device


def load_text(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download tinyshakespeare with:\n"
            "  wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt "
            "-O data/input.txt"
        )
    return path.read_text(encoding="utf-8")


@dataclass
class DataBundle:
    """Train/val token tensors plus a random-batch sampler."""

    train_data: torch.Tensor
    val_data: torch.Tensor
    tokenizer: CharTokenizer
    block_size: int
    batch_size: int
    device: torch.device

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def get_batch(self, split: str, generator: torch.Generator | None = None):
        """Return (x, y): x is (B, T) token ids, y is x shifted one step right."""
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,), generator=generator)
        offsets = torch.arange(self.block_size)
        x = data[ix[:, None] + offsets]
        y = data[ix[:, None] + offsets + 1]
        return x.to(self.device), y.to(self.device)


# --------------------------------------------------------------------- cache
def _cache_meta(text: str, tokenizer: CharTokenizer, train_split: float) -> dict:
    return {
        "chars": "".join(tokenizer.chars),
        "n_chars": len(text),
        "train_split": train_split,
        "sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
    }


def _load_cache(train_path: Path, val_path: Path, meta: dict):
    if not (train_path.exists() and val_path.exists()):
        return None
    try:
        tr = torch.load(train_path)
        va = torch.load(val_path)
    except Exception:
        return None
    if tr.get("meta") != meta or va.get("meta") != meta:
        return None
    return tr["data"], va["data"]


def prepare_data(cfg: Config, tokenizer: CharTokenizer | None = None,
                 device: torch.device | str | None = None) -> DataBundle:
    """Read `cfg.data_path`, tokenize it, split it, and return a `DataBundle`.

    Pass `tokenizer` to reuse a vocabulary (e.g. the one stored in a checkpoint).
    """
    text = load_text(cfg.resolve(cfg.data_path))
    tokenizer = tokenizer or CharTokenizer.from_text(text)
    device = get_device(cfg.device) if device is None else torch.device(device)

    meta = _cache_meta(text, tokenizer, cfg.train_split)
    cache_dir = cfg.resolve(cfg.cache_dir)
    train_path, val_path = cache_dir / "train.pt", cache_dir / "val.pt"

    cached = _load_cache(train_path, val_path, meta) if cfg.use_cache else None
    if cached is not None:
        train_data, val_data = cached
    else:
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(cfg.train_split * len(data))
        train_data, val_data = data[:n], data[n:]
        if cfg.use_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"data": train_data, "meta": meta}, train_path)
            torch.save({"data": val_data, "meta": meta}, val_path)

    for name, d in (("train", train_data), ("val", val_data)):
        if len(d) <= cfg.block_size + 1:
            raise ValueError(
                f"{name} split has {len(d)} tokens, which is too short for block_size={cfg.block_size}"
            )
    return DataBundle(train_data, val_data, tokenizer, cfg.block_size, cfg.batch_size, device)
