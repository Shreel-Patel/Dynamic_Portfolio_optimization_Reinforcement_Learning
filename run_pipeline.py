#!/usr/bin/env python3
"""Orchestrate fetch → preprocess → train → backtest (metrics + figures)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_step(desc: str, cmd: list[str]) -> None:
    print(f"\n>>> {desc}\n{' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full DRL portfolio pipeline")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument(
        "--agents",
        nargs="*",
        default=None,
        help="Agents for train/backtest (default: SAC DDPG PPO)",
    )
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_fetch:
        run_step("Fetch prices", [py, str(ROOT / "data" / "fetch.py")])

    if not args.skip_preprocess:
        run_step("Preprocess features", [py, str(ROOT / "data" / "preprocess.py")])

    agents = args.agents or ["SAC", "DDPG", "PPO"]
    train_cmd = [py, str(ROOT / "agents" / "train.py"), "--agents", *agents]

    if not args.skip_train:
        run_step("Train agents", train_cmd)

    if not args.skip_backtest:
        bt_cmd = [py, str(ROOT / "backtest" / "backtest.py"), "--agents", *agents]
        run_step("Backtest + metrics + plots", bt_cmd)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
