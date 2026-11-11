"""Central configuration for the nanoGPT project.

Every hyper-parameter lives in one dataclass (`Config`) so that a run is fully
described by a single object, which is also stored inside each checkpoint.
Named presets cover the common use cases:

    bigram   - the lecture's bigram baseline (a few seconds on a CPU)
    gpt      - the lecture's full GPT (~10.8M params; wants a GPU)
    gpt_cpu  - a scaled-down GPT that trains in minutes on a laptop CPU
    tiny     - a toy model used by the test-suite / smoke tests
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_TYPES = ("bigram", "gpt")
LR_SCHEDULES = ("cosine", "constant")
ATTN_IMPLS = ("fused", "loop")


@dataclass
class Config:
    # ---- run ---------------------------------------------------------------
    run_name: str = "gpt"
    seed: int = 1337
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"

    # ---- data --------------------------------------------------------------
    data_path: str = "data/input.txt"
    train_split: float = 0.9  # first 90% train, rest validation
    use_cache: bool = True  # cache encoded tensors in data/processed/
    cache_dir: str = "data/processed"

    # ---- model -------------------------------------------------------------
    model_type: str = "gpt"  # "bigram" | "gpt"
    block_size: int = 256  # maximum context length
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2
    attn_impl: str = "fused"  # "fused" (SDPA, fast) | "loop" (per-head, as in the lecture)

    # ---- optimisation ------------------------------------------------------
    batch_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    learning_rate: float = 3e-4
    min_lr: float = 3e-5  # floor of the cosine schedule
    warmup_iters: int = 100
    lr_schedule: str = "cosine"  # "cosine" | "constant"
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.99)
    grad_clip: float = 1.0  # 0 disables clipping
    grad_accum_steps: int = 1
    patience: int = 0  # early stopping: #evals without val improvement (0 = off)
    amp: bool = False  # bfloat16 autocast (CUDA only)

    # ---- outputs -----------------------------------------------------------
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "outputs/training_logs"
    save_checkpoints: bool = True
    save_logs: bool = True

    def __post_init__(self) -> None:
        self.betas = tuple(self.betas)
        self.validate()

    # ------------------------------------------------------------------ utils
    def validate(self) -> None:
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {self.model_type!r}")
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}, got {self.lr_schedule!r}")
        if self.attn_impl not in ATTN_IMPLS:
            raise ValueError(f"attn_impl must be one of {ATTN_IMPLS}, got {self.attn_impl!r}")
        if self.model_type == "gpt" and self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if not 0.0 < self.train_split < 1.0:
            raise ValueError("train_split must be in (0, 1)")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        for name in ("block_size", "batch_size", "eval_interval", "eval_iters", "grad_accum_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.max_iters < 0:
            raise ValueError("max_iters must be >= 0")

    def resolve(self, path: str) -> Path:
        """Resolve a (possibly relative) path against the project root."""
        p = Path(path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def with_overrides(self, **overrides) -> "Config":
        return replace(self, **overrides)


# --------------------------------------------------------------------------
# Presets (values are overrides on top of the dataclass defaults)
# --------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
    # Lecture bigram: batch 32, context 8, lr 1e-2, 3000 steps, constant LR.
    "bigram": dict(
        run_name="bigram", model_type="bigram", block_size=8, batch_size=32,
        max_iters=3000, eval_interval=300, eval_iters=200, learning_rate=1e-2,
        min_lr=1e-2, warmup_iters=0, lr_schedule="constant", weight_decay=0.01,
        grad_clip=0.0, dropout=0.0,
    ),
    # Lecture GPT: 6 layers x 6 heads x 384 dims, context 256 (~10.8M params).
    "gpt": dict(
        run_name="gpt", model_type="gpt", block_size=256, batch_size=64,
        n_embd=384, n_head=6, n_layer=6, dropout=0.2, max_iters=5000,
        eval_interval=500, learning_rate=3e-4, min_lr=3e-5, warmup_iters=100,
    ),
    # Small GPT for CPUs: ~0.8M params.
    "gpt_cpu": dict(
        run_name="gpt", model_type="gpt", block_size=64, batch_size=32,
        n_embd=128, n_head=4, n_layer=4, dropout=0.1, max_iters=2000,
        eval_interval=250, eval_iters=50, learning_rate=2e-3, min_lr=2e-4,
        warmup_iters=100,
    ),
    # Toy model for tests.
    "tiny": dict(
        run_name="tiny", model_type="gpt", block_size=16, batch_size=8,
        n_embd=32, n_head=2, n_layer=2, dropout=0.0, max_iters=30,
        eval_interval=10, eval_iters=5, learning_rate=3e-3, min_lr=3e-4,
        warmup_iters=3, save_checkpoints=False, save_logs=False, use_cache=False,
    ),
}


def get_config(preset: str = "gpt_cpu", **overrides) -> Config:
    """Build a `Config` from a named preset plus keyword overrides."""
    if preset not in PRESETS:
        raise KeyError(f"Unknown preset {preset!r}. Available: {sorted(PRESETS)}")
    return Config(**{**PRESETS[preset], **overrides})
