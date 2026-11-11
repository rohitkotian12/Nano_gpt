"""Character-level tokenizer (the lecture's `stoi` / `itos` maps, as a class)."""
from __future__ import annotations

from typing import Iterable


class CharTokenizer:
    """Maps every distinct character to an integer id and back."""

    def __init__(self, chars: Iterable[str]):
        chars = list(chars)
        if len(set(chars)) != len(chars):
            raise ValueError("Tokenizer characters must be unique")
        self.chars: list[str] = chars
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: dict[int, str] = {i: ch for i, ch in enumerate(chars)}

    # -------------------------------------------------------------- building
    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    # ------------------------------------------------------------ properties
    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def __len__(self) -> int:
        return self.vocab_size

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CharTokenizer) and self.chars == other.chars

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"

    # --------------------------------------------------------------- coding
    def unknown_chars(self, s: str) -> list[str]:
        return sorted({c for c in s if c not in self.stoi})

    def encode(self, s: str, errors: str = "strict") -> list[int]:
        """Encode a string. errors='strict' raises on unknown chars, 'ignore' drops them."""
        if errors == "ignore":
            return [self.stoi[c] for c in s if c in self.stoi]
        if errors != "strict":
            raise ValueError("errors must be 'strict' or 'ignore'")
        unknown = self.unknown_chars(s)
        if unknown:
            raise ValueError(f"Characters not in vocabulary: {unknown}")
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(self.itos[int(i)] for i in ids)

    # --------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        return {"chars": "".join(self.chars)}

    @classmethod
    def from_dict(cls, d: dict) -> "CharTokenizer":
        return cls(list(d["chars"]))
