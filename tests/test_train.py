import csv

import pytest
import torch

from configs.config import Config, get_config
from src.evaluate import estimate_loss, evaluate_checkpoint, loss_to_metrics
from src.data_loader import prepare_data
from src.generate import generate_text
from src.model import build_model, load_checkpoint
from src.train import get_lr, main as train_main, train


def test_lr_schedule_warmup_and_cosine():
    cfg = get_config("tiny", max_iters=100, warmup_iters=10, learning_rate=1e-3, min_lr=1e-4)
    assert get_lr(0, cfg) == pytest.approx(1e-4)          # (0+1)/10 * lr
    assert get_lr(9, cfg) == pytest.approx(1e-3)          # end of warmup
    assert get_lr(10, cfg) == pytest.approx(1e-3)         # cosine starts at the peak
    assert get_lr(55, cfg) == pytest.approx(5.5e-4, rel=1e-3)  # halfway
    assert get_lr(100, cfg) == pytest.approx(1e-4)        # floor at the end
    mid = [get_lr(i, cfg) for i in range(10, 100)]
    assert all(a >= b for a, b in zip(mid, mid[1:]))      # monotone decay


def test_constant_schedule():
    cfg = get_config("bigram")
    assert get_lr(0, cfg) == get_lr(2999, cfg) == 1e-2


def test_config_roundtrip_and_overrides():
    cfg = get_config("gpt_cpu", n_layer=3)
    assert Config.from_dict(cfg.to_dict()) == cfg
    assert cfg.n_layer == 3
    with pytest.raises(KeyError):
        get_config("nope")


def test_training_reduces_loss_and_writes_artifacts(tiny_cfg):
    cfg = tiny_cfg.with_overrides(max_iters=80, eval_interval=40, eval_iters=10, save_checkpoints=True,
                                  save_logs=True, learning_rate=5e-3, min_lr=5e-4, run_name="t")
    summary = train(cfg, verbose=False)
    assert summary["val_loss"] < 3.0  # random init is ~ln(vocab) ~ 2.7-4; this corpus is trivially learnable
    assert summary["best_val_loss"] <= summary["val_loss"] + 1e-9
    # artifacts
    assert (cfg.resolve(cfg.checkpoint_dir) / "t_best.pt").exists()
    with open(summary["log_path"]) as f:
        rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows] == [0, 40, 80]
    assert float(rows[-1]["train_loss"]) < float(rows[0]["train_loss"])


def test_checkpoint_roundtrip_gives_identical_logits(tiny_cfg):
    cfg = tiny_cfg.with_overrides(max_iters=20, eval_interval=10, eval_iters=3, save_checkpoints=True,
                                  run_name="rt")
    summary = train(cfg, verbose=False)
    ck = load_checkpoint(summary["best_checkpoint"])
    assert ck["config"].n_embd == cfg.n_embd
    x = torch.randint(0, ck["tokenizer"].vocab_size, (2, 8))
    ck2 = load_checkpoint(summary["best_checkpoint"])
    assert torch.equal(ck["model"](x)[0], ck2["model"](x)[0])


def test_resume_continues_from_saved_iteration(tiny_cfg):
    cfg = tiny_cfg.with_overrides(max_iters=20, eval_interval=10, eval_iters=3, save_checkpoints=True,
                                  save_logs=True, run_name="res")
    train(cfg.with_overrides(max_iters=10), verbose=False)
    train(cfg, resume=True, verbose=False)
    with open(cfg.resolve(cfg.log_dir) / "res.csv") as f:
        steps = [int(r["step"]) for r in csv.DictReader(f)]
    assert steps == [0, 10, 20]  # no duplicated eval at the resume point
    assert load_checkpoint(cfg.resolve(cfg.checkpoint_dir) / "res_last.pt")["iter_num"] == 20


def test_early_stopping(tiny_cfg):
    cfg = tiny_cfg.with_overrides(max_iters=200, eval_interval=5, eval_iters=3, patience=1,
                                  learning_rate=0.0, min_lr=0.0, warmup_iters=0, lr_schedule="constant")
    summary = train(cfg, verbose=False)  # lr = 0 -> val loss can't keep improving
    assert summary["stopped_early"]


def test_gradient_accumulation_runs(tiny_cfg):
    summary = train(tiny_cfg.with_overrides(grad_accum_steps=3, max_iters=10), verbose=False)
    assert summary["val_loss"] == summary["val_loss"]  # not NaN


def test_bigram_trains_and_generates(text_file, tmp_path):
    cfg = get_config("bigram", data_path=str(text_file), device="cpu", max_iters=300, eval_interval=150,
                     eval_iters=5, checkpoint_dir=str(tmp_path / "c"), log_dir=str(tmp_path / "l"),
                     cache_dir=str(tmp_path / "x"))
    summary = train(cfg, verbose=False)
    ck = load_checkpoint(summary["best_checkpoint"])
    text = generate_text(ck["model"], ck["tokenizer"], "All:", max_new_tokens=30, seed=0)
    assert text.startswith("All:") and len(text) == 34


def test_evaluate_checkpoint_is_deterministic(tiny_cfg):
    cfg = tiny_cfg.with_overrides(max_iters=10, eval_interval=10, eval_iters=3, save_checkpoints=True, run_name="ev")
    path = train(cfg, verbose=False)["best_checkpoint"]
    a = evaluate_checkpoint(path, eval_iters=5, device="cpu", seed=3)
    b = evaluate_checkpoint(path, eval_iters=5, device="cpu", seed=3)
    assert a["val"]["loss"] == b["val"]["loss"]
    assert a["val"]["perplexity"] == pytest.approx(torch.exp(torch.tensor(a["val"]["loss"])).item(), rel=1e-5)


def test_loss_to_metrics():
    m = loss_to_metrics(1.0)
    assert m["perplexity"] == pytest.approx(2.71828, rel=1e-4)
    assert m["bpc"] == pytest.approx(1.4427, rel=1e-4)


def test_cli_parses_overrides(tiny_cfg):
    summary = train_main(["--preset", "tiny", "--max_iters", "5", "--eval_interval", "5", "--eval_iters", "2",
                          "--data_path", tiny_cfg.data_path, "--device", "cpu", "--quiet", "--no-use_cache"])
    assert summary["val_loss"] == summary["val_loss"]


def test_generate_text_respects_seed(tiny_cfg):
    data = prepare_data(tiny_cfg)
    model = build_model(tiny_cfg, data.vocab_size)
    a = generate_text(model, data.tokenizer, "All:", 25, temperature=1.0, seed=9)
    b = generate_text(model, data.tokenizer, "All:", 25, temperature=1.0, seed=9)
    assert a == b and a.startswith("All:")
    # unknown prompt chars are dropped, empty prompts start from a newline
    assert generate_text(model, data.tokenizer, "@@@", 5, seed=1).startswith("\n")
