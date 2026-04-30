# Optional Makefile (requires GNU Make or compatible)

.PHONY: all fetch preprocess train backtest

all:
	python run_pipeline.py

fetch:
	python data/fetch.py

preprocess:
	python data/preprocess.py

train:
	python agents/train.py

backtest:
	python backtest/backtest.py
