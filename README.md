# nanoGPT: character-level language models from scratch

A clean, tested, modular PyTorch rebuild of Andrej Karpathy's
[nanogpt-lecture](https://github.com/karpathy/nanogpt-lecture) (the *Neural Networks: Zero to Hero* GPT lecture).
It trains a bigram baseline and a decoder-only Transformer on
[tinyshakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare), and ships with
a generation CLI, an evaluation CLI, a set of small ablation experiments, and a Streamlit playground.

The original lecture scripts are kept untouched in [`lecture/`](lecture/) so you can diff them against the
modular version.

## Results

| Model | Params | Context | Train time | Val loss | Perplexity | Bits / char |
|---|---|---|---|---|---|---|
| Uniform guess (65 chars) | - | - | - | 4.174 | 65.0 | 6.02 |
| Bigram (`--preset bigram`) | 4,225 | 8 | 1 s | 2.488 | 12.0 | 3.59 |
| GPT, small (`--preset gpt_cpu`) | ~0.82 M | 64 | 7 min (1 CPU thread) | **1.695** | **5.4** | **2.45** |

Validation numbers come from `python -m src.evaluate` (200 random batches, fixed seed). The small GPT is
the checkpoint shipped in `checkpoints/`. The full lecture-size model (`--preset gpt`, 10,788,929 parameters, the same
count as the original `gpt.py`) is configured but **was not trained here** because it needs a GPU.

Sample from the small GPT (`temperature=0.8, top_k=40`, full text in [`outputs/sample_generation.txt`](outputs/sample_generation.txt)):

```
ROMEO:
Why, here dear Edward, longs ears and
Thou speak that storm with this bany.

POLIXENES:
Ay, and it could your regocestands of the way.
```

For contrast, the bigram on the same prompt: `Cllan tcede. Y in, lroloundee waindonte ate t,-bou wist ...`.
The model learns the *shape* of a play (speaker names, line breaks, verse rhythm, mostly real words) but not meaning.

## Quick start

```bash
pip install -r requirements.txt

python -m src.train --preset bigram          # seconds
python -m src.train --preset gpt_cpu         # ~7 min on one CPU thread
python -m src.generate --checkpoint checkpoints/gpt_best.pt --prompt "ROMEO:" --temperature 0.8 --top_k 40
python -m src.evaluate --checkpoint checkpoints/gpt_best.pt
python -m streamlit run app/streamlit_app.py
python -m pytest
```

`make test`, `make gpt-cpu`, `make app`, etc. are shortcuts for the same commands.

Any config field can be overridden from the command line, for example:

```bash
python -m src.train --preset gpt_cpu --n_layer 6 --max_iters 5000 --run_name gpt_big
python -m src.train --preset gpt_cpu --resume                    # continue from checkpoints/gpt_last.pt
python -m src.train --preset gpt --attn_impl loop --amp          # full lecture config, per-head attention, bf16 on CUDA
```

## Project layout

```
nanoGPT-project/
├── configs/config.py        # one dataclass for every hyper-parameter + presets (bigram, gpt, gpt_cpu, tiny)
├── data/input.txt           # tinyshakespeare (1.1 MB); data/processed/ holds a gitignored tensor cache
├── src/
│   ├── tokenizer.py         # CharTokenizer (encode / decode / save vocabulary)
│   ├── data_loader.py       # 90/10 split, random batches, tensor cache
│   ├── bigram.py            # bigram baseline
│   ├── attention.py         # Head + MultiHeadAttention (lecture style) and fused CausalSelfAttention
│   ├── transformer.py       # FeedForward, Block
│   ├── model.py             # GPTLanguageModel, build_model, save/load checkpoint
│   ├── sampling.py          # temperature / top-k / top-p sampling, generate + generate_stream
│   ├── train.py             # training loop and CLI
│   ├── evaluate.py          # loss estimates, perplexity, bits-per-char
│   └── generate.py          # text generation CLI
├── app/streamlit_app.py     # playground: generation, training curves, experiments, model inspector
├── experiments/             # run_experiments.py + results.csv
├── outputs/                 # sample_generation.txt, training_logs/*.csv
├── checkpoints/             # bigram_best.pt, gpt_best.pt (tracked); *_last.pt (resume state, gitignored)
├── tests/                   # 44 tests
├── lecture/                 # original bigram.py, gpt.py and sample output from the lecture
├── requirements.txt  Makefile  pytest.ini  .gitignore
```

Checkpoints are self-contained: they store the weights, the full `Config` and the vocabulary, so generation
and the Streamlit app need nothing but the `.pt` file.

## How it was built (the 12 phases)

| Phase | What | Where |
|---|---|---|
| 1. Understand the code | Read `bigram.py` / `gpt.py`; notes below | `lecture/` |
| 2. Environment | Dependency list, Makefile | `requirements.txt`, `Makefile` |
| 3. Dataset | Character vocabulary (65 symbols), 90/10 split, batching | `tokenizer.py`, `data_loader.py` |
| 4. Bigram | Embedding-table baseline | `bigram.py` |
| 5. GPT | Causal self-attention, MLP, pre-norm residual blocks | `attention.py`, `transformer.py`, `model.py` |
| 6. Train and evaluate | Training loop, loss / perplexity / bpc | `train.py`, `evaluate.py` |
| 7. Modular code | One config object, small focused modules | `configs/`, `src/` |
| 8. Advanced training | AdamW param groups, warmup + cosine LR, grad clipping and accumulation, early stopping, resume, best/last checkpoints, CSV logs, bf16 autocast | `train.py` |
| 9. Better generation | Temperature, top-k, top-p, seeds, streaming | `sampling.py`, `generate.py` |
| 10. Experiments | 20 short ablation runs | `experiments/` |
| 11. Streamlit | Interactive playground | `app/` |
| 12. Final repo | Tests, README, `.gitignore` | this repo |

### Architecture (GPT)

Token embedding + learned position embedding → `n_layer` × [LayerNorm → causal multi-head attention → residual,
LayerNorm → MLP (4× wide, ReLU) → residual] → LayerNorm → linear head over the 65-character vocabulary.
Weights use the lecture's `N(0, 0.02)` init.

| Preset | n_layer | n_head | n_embd | Context | Batch | Iters | Peak LR |
|---|---|---|---|---|---|---|---|
| `gpt` (lecture) | 6 | 6 | 384 | 256 | 64 | 5000 | 3e-4 |
| `gpt_cpu` | 4 | 4 | 128 | 64 | 32 | 2000 | 2e-3 |

### Differences from the lecture code

- **Modular and configurable**, instead of globals in one file. Checkpoints carry their own config and vocabulary.
- **Two attention implementations.** `attn_impl="loop"` is the lecture's one-`Head`-per-head version;
  `"fused"` (default) does all heads in one batched `scaled_dot_product_attention` call. A test copies weights
  between them and checks the outputs match, and both have the same parameter count.
- **Logits are always `(B, T, V)`.** The lecture flattens them to `(B*T, V)` whenever targets are passed.
- **Generation** crops the context to `block_size` for every model (the lecture's bigram let it grow), switches
  to eval mode so dropout is off while sampling, and supports temperature, top-k, top-p and a seed.
- **Position embeddings** use `idx.device` rather than a global `device`.
- **Training:** weight decay only on matrices, LR warmup + cosine decay, gradient clipping, and the final
  evaluation happens *after* the last update (the lecture evaluates just before it).

## Experiments

`python -m experiments.run_experiments` trains 20 runs and appends them to
[`experiments/results.csv`](experiments/results.csv). Each GPT run starts from one small base model (2 layers,
4 heads, 64 dims, context 64, no dropout, LR 3e-3 with cosine decay, **400 iterations**, seed 1337) and changes one
thing. Validation loss is averaged over 50 random batches.

| Question | Runs (val loss) | Reading |
|---|---|---|
| **Run-to-run noise** | base 2.151 · seed 1: 2.166 · seed 2: 2.176 | Identical configs differ by about ±0.02, so smaller gaps than ~0.05 are not meaningful |
| Width | 32: 2.277 · 64: 2.151 · 128: **1.952** | Clear win for width |
| Learning rate | 3e-4: 2.579 · 1e-3: 2.333 · 3e-3: 2.151 · 1e-2: **2.060** | Higher is better in 400 steps; 1e-2 is untested on longer or larger runs |
| Context length | 8: 2.316 · 16: 2.186 · 32: 2.157 · 64: 2.151 | Big gain from 8 to 16, then flat |
| Dropout | 0.0: 2.151 · 0.2: 2.235 | Hurts at this short horizon |
| Depth | 1: 2.122 · 2: 2.151 · 4: 2.185 | Within noise |
| Heads | 1: 2.091 · 2: 2.110 · 4: 2.151 · 8: 2.145 | Within noise |
| LR schedule | cosine: 2.151 · constant: 2.105 | Borderline; cosine hasn't paid off in so few steps |
| Attention impl. | fused: 2.151 in 15.8 s · loop: 2.133 in 21.0 s | Same architecture (the loss gap is init randomness); fused ~1.3× faster here |
| Model family | bigram: 2.498 · GPT base: 2.151 | The Transformer wins by a wide margin |

**Caveats.** Single seed per configuration, short runs on a small model, and equal *iterations* rather than
equal compute (shorter contexts see fewer tokens per step). Use these results for relative comparisons only; effects such as
dropout and depth typically change sign with longer training. Timings are from one CPU thread and
are indicative only. The full table is in the CSV and in the app's *Experiments* tab.

## Testing

`python -m pytest` runs 44 tests in about 4 seconds on a CPU: tokenizer round-trips, batch shapes and shifting,
cache invalidation, initial loss ≈ ln(vocab), **causality** (changing future tokens leaves earlier logits
unchanged, for both attention versions), fused-vs-loop equivalence, top-k / top-p behaviour, seed reproducibility,
LR schedule values, loss reduction during training, checkpoint round-trip, resume without duplicate evaluations,
early stopping, gradient accumulation, and the CLI.

Verified with Python 3.12, PyTorch 2.14 (CPU build) and Streamlit 1.67. The CUDA and bf16 (`--amp`) code paths
are small but were **not** exercised, since no GPU was available.

## Limitations and next steps

- Character-level modelling is slow to learn long-range structure; a BPE tokenizer would help.
- No KV cache, so generation recomputes the whole context at every step.
- Possible extensions: weight tying, GELU, scaled residual init, `torch.compile`, a larger multi-seed sweep.

## Credits and license

Model code and the lecture design come from [Andrej Karpathy's nanogpt-lecture](https://github.com/karpathy/nanogpt-lecture)
and the *Zero to Hero* series (MIT licensed). If you publish this repo, add a `LICENSE` file that keeps his
attribution. tinyshakespeare is from [char-rnn](https://github.com/karpathy/char-rnn).
