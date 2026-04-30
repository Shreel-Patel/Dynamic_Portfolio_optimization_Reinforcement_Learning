# Deep RL for Portfolio Optimization (FE 529-B)

This repository implements **DDPG**, **SAC**, and **PPO** on a custom Gymnasium portfolio environment with proportional transaction costs (10 bps on turnover), plus **Buy & Hold**, **Equal-Weight**, and **rolling Markowitz (max Sharpe)** benchmarks.

## Setup

```bash
cd drl_portfolio
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

## Run end-to-end

```bash
python run_pipeline.py
```

Stages:

1. `data/fetch.py` — download adjusted closes (yfinance) to `data/raw/adj_close.csv`
2. `data/preprocess.py` — features + rolling z-scores → `data/processed/features_returns.csv`
3. `agents/train.py` — trains SAC, DDPG, PPO; TensorBoard logs under `results/logs/`; checkpoints `results/models/{AGENT}_best.zip` / `_final.zip`
4. `backtest/backtest.py` — test-window evaluation, `results/metrics_comparison.csv`, figures under `results/figures/`

### Partial runs

```bash
python run_pipeline.py --skip-train --skip-fetch    # preprocess + backtest only (needs models)
python run_pipeline.py --skip-fetch --skip-preprocess
python agents/train.py --agents SAC DDPG
python backtest/backtest.py --agents SAC --final      # use *_final.zip if best missing
```

### TensorBoard

```bash
tensorboard --logdir results/logs
```

### Hyperparameter tuning (stub)

```bash
python agents/hyperopt.py --agent SAC --trials 50
```

Wire `optuna` objectives to `agents/train.py` per course rubric.

## Notebooks

Optional Jupyter workflows live under `notebooks/` (`01_data_prep.ipynb`, `02_train_agents.ipynb`, `03_analysis.ipynb`). They mirror the CLI pipeline.

## Configuration

Edit `config/config.yaml` for tickers, date splits, transaction cost, agent defaults, and training budgets.

## Validation checks

- Features use rolling statistics shifted by one bar (`preprocess.rolling_zscore`) to limit look-ahead bias.
- Train / validation / test indices are disjoint (`data/preprocess.split_indices`).
- Environment applies turnover costs before updating weights (`environments/portfolio_env.py`).

## Project layout

See `plan.md` for the specification used to generate this codebase.

## Publish to GitHub

Create an empty repository on GitHub (for example: `drl-portfolio-fe529`), then run:

```bash
git init
git add .
git commit -m "Initial commit: DRL portfolio optimization pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
