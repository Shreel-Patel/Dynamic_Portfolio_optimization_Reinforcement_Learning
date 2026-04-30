# Project Plan: Deep RL for Portfolio Optimization with Transaction Costs

This file mirrors the course specification (overview, stack, splits, environment, agents, benchmarks, deliverables). Implementation lives in the sibling modules (`environments/`, `agents/`, `backtest/`, `evaluation/`, `data/`).

## Highlights

- Tickers: AAPL, MSFT, GOOGL, AMZN, NVDA — daily data 2020–2024 via `yfinance`.
- Train / validation / test: through 2022 / Q1–Q3 2023 / Oct 2023–2024 (see `config/config.yaml`).
- RL algorithms: SAC, DDPG, PPO via Stable-Baselines3; continuous actions; reward = log portfolio return after costs.

Refer to the assignment PDF for report length, literature review, and submission requirements.
