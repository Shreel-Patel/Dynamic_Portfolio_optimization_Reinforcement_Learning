"""Evaluate trained agents on the held-out test window; combine with benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import DDPG, PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.train import load_config, make_vec_env_fn  # noqa: E402
from backtest.benchmarks import benchmark_suite  # noqa: E402
from data.preprocess import dataframe_to_tensors, split_indices  # noqa: E402
from environments.portfolio_env import PortfolioEnv  # noqa: E402
from evaluation.metrics import metrics_table  # noqa: E402
from evaluation.plots import (  # noqa: E402
    plot_drawdowns,
    plot_portfolio_performance,
    plot_rolling_sharpe,
    plot_weights_heatmap,
    set_style,
)


def load_model_for_agent(name: str, path: Path, env):
    name_u = name.upper()
    if name_u == "SAC":
        return SAC.load(path, env=env)
    if name_u == "DDPG":
        return DDPG.load(path, env=env)
    if name_u == "PPO":
        return PPO.load(path, env=env)
    raise ValueError(name)


def rollout_returns_and_weights(
    model,
    feat: np.ndarray,
    ret: np.ndarray,
    lookback: int,
    tc: float,
    seed: int,
    date_index: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.DataFrame | None]:
    """Deterministic rollout on test tensors; align daily returns to calendar dates."""

    def _thunk():
        return Monitor(PortfolioEnv(feat, ret, lookback, transaction_cost=tc, seed=seed))

    vec = DummyVecEnv([_thunk])
    reset_out = vec.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    values: list[float] = [1.0]
    weights_rows: list[np.ndarray] = []

    for _ in range(500000):
        action, _ = model.predict(obs, deterministic=True)
        step_out = vec.step(action)
        if len(step_out) == 5:
            obs, _, done, _, infos = step_out
        else:
            obs, _, done, infos = step_out
        info_list = infos if isinstance(infos, (list, tuple)) else [infos]
        info = info_list[0]
        values.append(float(info["portfolio_value"]))
        w = np.asarray(info.get("weights_risky", []), dtype=float).reshape(-1)
        weights_rows.append(w)
        if np.asarray(done).reshape(-1)[0]:
            break

    v = np.asarray(values, dtype=float)
    daily_simple = np.diff(v) / np.maximum(v[:-1], 1e-12)

    # Env timestep t uses return[risky] at row t; first transition aligns to date_index[lookback+1]
    if len(daily_simple) == 0:
        return pd.Series(dtype=float), None
    ix_start = lookback + 1
    ix_end = ix_start + len(daily_simple)
    idx_slice = date_index[ix_start:ix_end]
    n = min(len(idx_slice), len(daily_simple))
    s = pd.Series(daily_simple[:n], index=idx_slice[:n])

    wdf = None
    if weights_rows and len(weights_rows) == len(daily_simple):
        wdf = pd.DataFrame(weights_rows[:n], index=idx_slice[:n])
    return s, wdf


def run_backtest(
    cfg_path: Path | None = None,
    use_best: bool = True,
    agents: list[str] | None = None,
) -> pd.DataFrame:
    cfg = load_config(cfg_path)
    agents = agents or ["SAC", "DDPG", "PPO"]

    data_csv = ROOT / "data" / "processed" / "features_returns.csv"
    raw_csv = ROOT / "data" / "raw" / "adj_close.csv"
    if not data_csv.exists():
        raise FileNotFoundError(data_csv)
    prices = pd.read_csv(raw_csv, index_col=0, parse_dates=True).sort_index()

    df = pd.read_csv(data_csv, index_col=0, parse_dates=True)
    tickers = cfg["tickers"]
    feat_full, ret_full = dataframe_to_tensors(df, tickers)

    train_idx, _val_idx, test_idx = split_indices(df.index, cfg)
    test_start_date = pd.Timestamp(cfg["data"]["test_start"])
    prices_test = prices.loc[prices.index >= test_start_date]

    bench_cfg = {
        "risk_free_annual": cfg["environment"]["risk_free_annual"],
        "markowitz_lookback": cfg["benchmarks"]["markowitz_lookback"],
        "equal_weight_freq": cfg["benchmarks"]["equal_weight_freq"],
    }
    bench_full = benchmark_suite(prices, bench_cfg)
    bench_test = bench_full.loc[bench_full.index >= test_start_date].dropna(how="any")

    lookback = int(cfg["environment"]["lookback"])
    tc = float(cfg["environment"]["transaction_cost"])

    feat_test = feat_full[test_idx]
    ret_test = ret_full[test_idx]
    dates_test = df.index[test_idx]

    strategies: dict[str, pd.Series] = {}
    weight_hist: dict[str, pd.DataFrame] = {}

    models_dir = ROOT / "results" / "models"

    for name in agents:
        suffix = "best" if use_best else "final"
        zp = models_dir / f"{name.upper()}_{suffix}.zip"
        if not zp.exists():
            zp = models_dir / f"{name.upper()}_final.zip"
        if not zp.exists():
            print(f"Skip {name}: missing model zip under {models_dir}")
            continue

        dummy_vec = DummyVecEnv(
            [
                make_vec_env_fn(
                    feat_test,
                    ret_test,
                    lookback,
                    tc,
                    cfg["training"]["seed"],
                )
            ]
        )
        model = load_model_for_agent(name, zp, dummy_vec)

        aligned, wdf = rollout_returns_and_weights(
            model,
            feat_test,
            ret_test,
            lookback,
            tc,
            cfg["training"]["seed"] + 7,
            dates_test,
        )
        strategies[name.upper()] = aligned
        if wdf is not None:
            weight_hist[name.upper()] = wdf

    for col in bench_test.columns:
        strategies[f"Bench_{col}"] = bench_test[col]

    set_style()
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if strategies:
        plot_portfolio_performance(strategies, fig_dir / "portfolio_performance.png")
        plot_drawdowns(strategies, fig_dir / "drawdowns.png")
        plot_rolling_sharpe(
            strategies,
            fig_dir / "rolling_sharpe.png",
            window=int(cfg["plots"]["rolling_sharpe_window"]),
            risk_free_annual=float(cfg["environment"]["risk_free_annual"]),
        )

    if weight_hist:
        best_agent = max(weight_hist.keys(), key=lambda k: len(weight_hist[k]))
        wh = weight_hist[best_agent]
        wh.columns = tickers[: wh.shape[1]]
        plot_weights_heatmap(wh.T, fig_dir / "weights_evolution.png", title=f"Weights ({best_agent})")

    table = metrics_table(
        strategies,
        risk_free_annual=float(cfg["environment"]["risk_free_annual"]),
    )
    out_csv = ROOT / "results" / "metrics_comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv)
    print(f"Wrote {out_csv}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--final", action="store_true", help="Use *_final.zip instead of *_best.zip")
    parser.add_argument("--agents", nargs="*", default=None)
    args = parser.parse_args()
    run_backtest(cfg_path=args.config, use_best=not args.final, agents=args.agents)


if __name__ == "__main__":
    main()
