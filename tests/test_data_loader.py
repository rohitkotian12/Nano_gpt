import pytest
import torch

from src.data_loader import load_text, prepare_data


def test_split_sizes(tiny_cfg, text_file):
    data = prepare_data(tiny_cfg)
    total = len(text_file.read_text())
    assert len(data.train_data) == int(0.9 * total)
    assert len(data.train_data) + len(data.val_data) == total


def test_batch_shapes_and_shift(tiny_cfg):
    data = prepare_data(tiny_cfg)
    x, y = data.get_batch("train")
    assert x.shape == y.shape == (tiny_cfg.batch_size, tiny_cfg.block_size)
    assert x.dtype == torch.long
    # y is x shifted by one position
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_batches_are_reproducible_with_generator(tiny_cfg):
    data = prepare_data(tiny_cfg)
    a = data.get_batch("val", torch.Generator().manual_seed(7))
    b = data.get_batch("val", torch.Generator().manual_seed(7))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_invalid_split(tiny_cfg):
    with pytest.raises(ValueError):
        prepare_data(tiny_cfg).get_batch("test")


def test_missing_file(tiny_cfg):
    with pytest.raises(FileNotFoundError):
        load_text("/nonexistent/input.txt")


def test_cache_roundtrip_and_invalidation(tiny_cfg, text_file):
    cfg = tiny_cfg.with_overrides(use_cache=True)
    first = prepare_data(cfg)
    assert (cfg.resolve(cfg.cache_dir) / "train.pt").exists()
    second = prepare_data(cfg)  # served from the cache
    assert torch.equal(first.train_data, second.train_data)
    # changing the text must invalidate the cache
    text_file.write_text(text_file.read_text() + "extra tail text\n" * 5)
    third = prepare_data(cfg)
    assert len(third.train_data) + len(third.val_data) == len(text_file.read_text())


def test_too_short_corpus_raises(tmp_path, tiny_cfg):
    p = tmp_path / "short.txt"
    p.write_text("abc")
    with pytest.raises(ValueError):
        prepare_data(tiny_cfg.with_overrides(data_path=str(p)))
