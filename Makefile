.PHONY: install test bigram gpt-cpu gpt generate evaluate experiments app

install:
	pip install -r requirements.txt

test:
	python -m pytest

bigram:            ## baseline, a few seconds on a CPU
	python -m src.train --preset bigram

gpt-cpu:           ## small GPT, ~15 min on a laptop CPU
	python -m src.train --preset gpt_cpu

gpt:               ## full lecture config (~10.8M params), GPU recommended
	python -m src.train --preset gpt

generate:
	python -m src.generate --checkpoint checkpoints/gpt_best.pt --prompt "ROMEO:" --temperature 0.8 --top_k 40

evaluate:
	python -m src.evaluate --checkpoint checkpoints/gpt_best.pt

experiments:
	python -m experiments.run_experiments

app:
	python -m streamlit run app/streamlit_app.py
