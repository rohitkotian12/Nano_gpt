import pytest
import torch

from src.tokenizer import CharTokenizer


def test_roundtrip():
    tok = CharTokenizer.from_text("hello world\n")
    s = "held\nlow"
    assert tok.decode(tok.encode(s)) == s


def test_vocab_is_sorted_and_unique():
    tok = CharTokenizer.from_text("banana")
    assert tok.chars == ["a", "b", "n"]
    assert tok.vocab_size == 3 == len(tok)


def test_unknown_chars_strict_and_ignore():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError):
        tok.encode("abz")
    assert tok.encode("abz", errors="ignore") == [0, 1]
    assert tok.unknown_chars("abzq") == ["q", "z"]


def test_decode_accepts_tensors():
    tok = CharTokenizer.from_text("abc")
    assert tok.decode(torch.tensor([2, 1, 0])) == "cba"


def test_serialisation_roundtrip():
    tok = CharTokenizer.from_text("some text\n")
    assert CharTokenizer.from_dict(tok.to_dict()) == tok


def test_duplicate_chars_rejected():
    with pytest.raises(ValueError):
        CharTokenizer(["a", "a"])
