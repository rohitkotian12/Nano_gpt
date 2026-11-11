# NanoGPT: Character-Level Language Models from Scratch

A modular, tested PyTorch implementation of character-level language models

This project implements and compares two language-model architectures: a Bigram model and a decoder-only GPT Transformer. It uses the Tiny Shakespeare dataset to train models capable of generating Shakespeare-style text.

The project also includes training and evaluation pipelines, configurable text generation, ablation experiments, automated testing, and an interactive Streamlit playground.

## Features

- Character-level tokenization with a 65-character vocabulary.
- Bigram language model as a baseline.
- Decoder-only GPT Transformer architecture.
- Causal multi-head self-attention.
- Fused and loop-based attention implementations.
- Configurable training presets.
- AdamW optimization and learning-rate scheduling.
- Gradient clipping and accumulation.
- Checkpoint saving and training resumption.
- Text generation using Temperature, Top-K and Top-P sampling.
- Model evaluation using validation loss, perplexity and bits per character.
- Ablation experiments.
- Interactive Streamlit application.
- Automated Pytest test suite.

## Model Performance

The following results are documented from the project's existing experiments.

| Model | Parameters | Validation Loss | Perplexity |
|---|---:|---:|---:|
| Uniform Guess | - | 4.174 | 65.0 |
| Bigram | 4,225 | 2.488 | 12.0 |
| Small GPT | ~0.82M | **1.695** | **5.4** |

The small GPT model performs better than the Bigram baseline on the reported validation metrics.

*Note: The full lecture-size GPT configuration is available but was not trained for these benchmark results.*

## Architecture

The GPT model follows a decoder-only Transformer architecture.

Input Characters → Token Embeddings → Positional Embeddings → Transformer Blocks → Layer Normalization → Linear Language Model Head → Next Character Prediction.

Each Transformer block contains:

1. Layer Normalization.
2. Causal Multi-Head Self-Attention.
3. Residual Connection.
4. Layer Normalization.
5. Feed-Forward Network.
6. Residual Connection.

## Technology Stack

- **Programming Language:** Python
- **Deep Learning Framework:** PyTorch
- **Data Processing:** Pandas
- **Interactive Interface:** Streamlit
- **Testing:** Pytest
- **Model Architecture:** Transformer / GPT
- **Dataset:** Tiny Shakespeare

## Project Structure

```text
nanoGPT-project/
├── app/
│   └── streamlit_app.py
├── checkpoints/
│   ├── bigram_best.pt
│   └── gpt_best.pt
├── configs/
│   └── config.py
├── data/
│   ├── input.txt
│   └── processed/
├── experiments/
│   ├── run_experiments.py
│   └── results.csv
├── lecture/
│   ├── bigram.py
│   └── gpt.py
├── outputs/
│   ├── sample_generation.txt
│   └── training_logs/
├── src/
│   ├── tokenizer.py
│   ├── data_loader.py
│   ├── bigram.py
│   ├── attention.py
│   ├── transformer.py
│   ├── model.py
│   ├── sampling.py
│   ├── train.py
│   ├── evaluate.py
│   └── generate.py
├── tests/
├── .gitignore
├── conftest.py
├── Makefile
├── pytest.ini
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd nanoGPT-project
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

Run all commands from the project root directory.

### 1. Train the Bigram Model

```bash
python -m src.train --preset bigram
```

### 2. Train the GPT Model (CPU)

```bash
python -m src.train --preset gpt_cpu
```

### 3. Train the Full GPT Model

```bash
python -m src.train --preset gpt
```

A CUDA-compatible GPU is recommended for the full configuration.

### 4. Generate Text

```bash
python -m src.generate --checkpoint checkpoints/gpt_best.pt --prompt "ROMEO:" --temperature 0.8 --top_k 40
```

### 5. Evaluate the Model

```bash
python -m src.evaluate --checkpoint checkpoints/gpt_best.pt
```

### 6. Launch the Interactive Playground

```bash
python -m streamlit run app/streamlit_app.py
```

### 7. Run Experiments

```bash
python -m experiments.run_experiments
```

### 8. Run Automated Tests

```bash
python -m pytest
```

## Experiments and Evaluation

The project includes ablation experiments to investigate the effects of different hyperparameters and architectural choices.

The experiments investigate:

- Embedding dimensions.
- Learning rates.
- Context lengths.
- Dropout.
- Transformer depth.
- Number of attention heads.
- Learning-rate scheduling.
- Attention implementations.
- Bigram versus GPT architecture.

Experiment results are stored in `experiments/results.csv`.

## Limitations

- Character-level tokenization is less efficient than subword tokenization for longer text.
- The model is trained on a relatively small dataset.
- Generated text may contain grammatically or semantically incorrect sentences.
- Generation does not currently implement KV caching.
- The full GPT configuration requires more computational resources.
- CUDA and BF16 training paths have not been validated in the documented CPU environment.

## Future Improvements

- Implement Byte Pair Encoding (BPE) tokenization.
- Add KV caching for faster text generation.
- Experiment with GELU activation and weight tying.
- Improve model architecture and initialization.
- Introduce larger datasets.
- Conduct multi-seed experiments.
- Optimize training using GPU acceleration and `torch.compile`.
---

*Built as an educational implementation to understand, train, evaluate, and experiment with GPT-style language models from scratch.*
