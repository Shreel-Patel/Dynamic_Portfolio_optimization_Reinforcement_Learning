"""Comparison plots for portfolio strategies."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def cumulative_wealth(daily_returns: pd.Series) -> pd.Series:
    return (1.0 + daily_returns.fillna(0.0)).cumprod()


def plot_portfolio_performance(
    strategies: dict[str, pd.Series],
    out_path: Path,
    log_scale: bool = True,
    title: str = "Portfolio value (normalized)",
) -> None:
    plt.figure(figsize=(11, 6))
    for name, r in strategies.items():
        w = cumulative_wealth(r)
        plt.plot(w.index, w.values, label=name, linewidth=1.5)
    plt.title(title)
    plt.ylabel("Wealth")
    if log_scale:
        plt.yscale("log")
    plt.xlabel("Date")
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_drawdowns(
    strategies: dict[str, pd.Series],
    out_path: Path,
    title: str = "Drawdown",
) -> None:
    plt.figure(figsize=(11, 5))
    for name, r in strategies.items():
        wealth = cumulative_wealth(r)
        peak = wealth.cummax()
        dd = wealth / peak - 1.0
        plt.plot(dd.index, dd.values, label=name, linewidth=1.2)
    plt.title(title)
    plt.ylabel("Drawdown")
    plt.xlabel("Date")
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_rolling_sharpe(
    strategies: dict[str, pd.Series],
    out_path: Path,
    window: int = 90,
    risk_free_annual: float = 0.02,
) -> None:
    rf_d = (1.0 + risk_free_annual) ** (1.0 / 252.0) - 1.0
    plt.figure(figsize=(11, 5))
    for name, r in strategies.items():
        xs = r.dropna().astype(float) - rf_d
        roll = xs.rolling(window).mean() / xs.rolling(window).std() * np.sqrt(252.0)
        plt.plot(roll.index, roll.values, label=name, linewidth=1.2)
    plt.axhline(0, color="gray", lw=0.8)
    plt.title(f"Rolling Sharpe ({window}-day)")
    plt.ylabel("Sharpe")
    plt.xlabel("Date")
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_weights_heatmap(weights: pd.DataFrame, out_path: Path, title: str = "Weights") -> None:
    plt.figure(figsize=(11, 5))
    sns.heatmap(weights.T, cmap="viridis", cbar=True)
    plt.title(title)
    plt.ylabel("Asset")
    plt.xlabel("Time")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
