"""
Optional Optuna hyperparameter search (extra credit).

Example objective: maximize validation Sharpe by tuning learning_rate, gamma, batch_size.

Usage (after implementing objective wiring to agents.train):
  python agents/hyperopt.py --agent SAC --trials 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="SAC")
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    try:
        import optuna  # noqa: F401
    except ImportError:
        raise SystemExit("Install optuna: pip install optuna")

    print(
        "Stub: wire Optuna study to train_one_agent(config_override) "
        f"for agent={args.agent} with n_trials={args.trials}. "
        "See README optional section."
    )


if __name__ == "__main__":
    main()
