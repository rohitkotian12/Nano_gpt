import pytest

from configs.config import get_config

SAMPLE_TEXT = ("First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
               "All:\nSpeak, speak.\n\n") * 40


@pytest.fixture
def text_file(tmp_path):
    p = tmp_path / "tiny.txt"
    p.write_text(SAMPLE_TEXT, encoding="utf-8")
    return p


@pytest.fixture
def tiny_cfg(text_file, tmp_path):
    """A very small config pointing at a throw-away corpus and temp output dirs."""
    return get_config(
        "tiny", data_path=str(text_file), device="cpu", checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "logs"), cache_dir=str(tmp_path / "cache"),
    )
