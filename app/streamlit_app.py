"""Interactive playground for the trained models.

Run from the project root:
    python -m streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.generate import stream_text  # noqa: E402
from src.model import load_checkpoint  # noqa: E402
from src.utils import count_parameters, format_params, get_device  # noqa: E402

CKPT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "outputs" / "training_logs"
RESULTS_CSV = ROOT / "experiments" / "results.csv"

st.set_page_config(page_title="nanoGPT playground", page_icon="🎭", layout="wide")


@st.cache_resource(show_spinner="Loading checkpoint...")
def load_model(path: str, mtime: float):
    """Cached per (path, mtime) so retraining a model invalidates the cache."""
    return load_checkpoint(path, get_device("auto"))


# ------------------------------------------------------------------ sidebar
checkpoints = sorted(CKPT_DIR.glob("*_best.pt"))
st.sidebar.title("🎭 nanoGPT")
if not checkpoints:
    st.error("No checkpoints found in `checkpoints/`. Train a model first, e.g.\n\n"
             "`python -m src.train --preset bigram` or `python -m src.train --preset gpt_cpu`")
    st.stop()

names = [p.stem.removesuffix("_best") for p in checkpoints]
default_idx = names.index("gpt") if "gpt" in names else 0
choice = st.sidebar.selectbox("Model", names, index=default_idx)
ckpt_path = checkpoints[names.index(choice)]
ckpt = load_model(str(ckpt_path), ckpt_path.stat().st_mtime)
model, tok, cfg = ckpt["model"], ckpt["tokenizer"], ckpt["config"]

st.sidebar.caption(
    f"{cfg.model_type} · {format_params(count_parameters(model))} params · "
    f"{ckpt['iter_num']} iters · best val loss {ckpt['best_val_loss']:.3f}"
)

st.sidebar.subheader("Sampling")
max_new = st.sidebar.slider("Characters to generate", 50, 2000, 400, step=50)
temperature = st.sidebar.slider("Temperature (0 = greedy)", 0.0, 2.0, 0.8, step=0.05,
                                help="Lower = safer and more repetitive, higher = more surprising.")
top_k = st.sidebar.slider("Top-k (0 = off)", 0, tok.vocab_size, 0,
                          help="Only sample from the k most likely characters.")
top_p = st.sidebar.slider("Top-p (1.0 = off)", 0.05, 1.0, 1.0, step=0.05,
                          help="Nucleus sampling: smallest set of characters with total probability >= p.")
fixed_seed = st.sidebar.checkbox("Fixed seed", value=True)
seed = st.sidebar.number_input("Seed", value=1337, step=1, disabled=not fixed_seed)

# --------------------------------------------------------------------- tabs
tab_gen, tab_curves, tab_exp, tab_model = st.tabs(["✍️ Generate", "📉 Training curves", "🧪 Experiments", "🔍 Model"])

with tab_gen:
    st.header("Generate Shakespeare-ish text")
    st.caption("The model is character-level: it learned tinyshakespeare one character at a time, "
               "so expect Shakespearean *shapes* rather than meaning.")
    prompt = st.text_area("Prompt (optional)", value="ROMEO:\n", height=100)
    dropped = tok.unknown_chars(prompt)
    if dropped:
        st.warning(f"These characters are not in the model's vocabulary and will be ignored: {dropped}")

    if st.button("Generate", type="primary"):
        slot = st.empty()
        text = ""
        kwargs = dict(max_new_tokens=max_new, temperature=temperature,
                      top_k=top_k or None, top_p=top_p if top_p < 1.0 else None,
                      seed=int(seed) if fixed_seed else None)
        for i, piece in enumerate(stream_text(model, tok, prompt, **kwargs)):
            text += piece
            if i % 5 == 0:  # refreshing the page on every character is needlessly slow
                slot.text(text)
        slot.text(text)
        st.session_state["last_output"] = text

    if st.session_state.get("last_output"):
        st.download_button("Download text", st.session_state["last_output"], file_name="generation.txt")

with tab_curves:
    st.header("Training curves")
    logs = sorted(LOG_DIR.glob("*.csv"))
    if not logs:
        st.info("No training logs yet. They are written to `outputs/training_logs/` during training.")
    else:
        log_names = [p.stem for p in logs]
        sel = st.selectbox("Run", log_names, index=log_names.index(choice) if choice in log_names else 0)
        df = pd.read_csv(LOG_DIR / f"{sel}.csv")
        c1, c2, c3 = st.columns(3)
        c1.metric("Best val loss", f"{df['val_loss'].min():.4f}")
        c2.metric("Final train loss", f"{df['train_loss'].iloc[-1]:.4f}")
        c3.metric("Training time", f"{df['elapsed_s'].iloc[-1]:.0f}s")
        st.line_chart(df.set_index("step")[["train_loss", "val_loss"]])
        with st.expander("Raw log"):
            st.dataframe(df)

with tab_exp:
    st.header("Experiments")
    if not RESULTS_CSV.exists():
        st.info("Run `python -m experiments.run_experiments` to produce `experiments/results.csv`.")
    else:
        res = pd.read_csv(RESULTS_CSV)
        groups = sorted(res["group"].unique())
        picked = st.multiselect("Groups", groups, default=groups)
        view = res[res["group"].isin(picked)].sort_values("val_loss")
        st.bar_chart(view.set_index("experiment")["val_loss"])
        st.dataframe(view, hide_index=True)
        st.caption("Short, equal-iteration runs on a small base model: good for relative comparisons, "
                   "not for final quality.")

with tab_model:
    st.header(f"Model: {choice}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Parameters", format_params(count_parameters(model)))
    c2.metric("Vocabulary", tok.vocab_size)
    c3.metric("Context length", cfg.block_size)
    st.subheader("Training config")
    st.json(cfg.to_dict(), expanded=False)
    st.subheader("Architecture")
    st.code(str(model), language="text")
    st.subheader("Vocabulary")
    st.code(repr("".join(tok.chars)), language="text")
