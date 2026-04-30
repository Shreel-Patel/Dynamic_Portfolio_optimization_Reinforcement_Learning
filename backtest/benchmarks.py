"""Buy & Hold, equal-weight, and rolling Markowitz (max Sharpe) benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _daily_rf(risk_free_annual: float = 0.02) -> float:
    return (1.0 + risk_free_annual) ** (1.0 / 252.0) - 1.0


def buy_and_hold_equal_initial(prices: pd.DataFrame) -> pd.Series:
    """Equal dollar allocation at first bar, no rebalancing."""
    px = prices.astype(float).ffill().dropna(how="any")
    r = px.pct_change().dropna()
    n = r.shape[1]
    w = np.ones(n, dtype=float) / n
    port_ret = (r.values * w.reshape(1, -1)).sum(axis=1)
    return pd.Series(port_ret, index=r.index, name="buy_hold_eq")


def equal_weight_monthly(prices: pd.DataFrame, freq: str = "ME") -> pd.Series:
    """Rebalance to 1/n at month-end closes."""
    del freq  # API compatibility — calendar month-end on index
    px = prices.astype(float).ffill().dropna(how="any")
    r = px.pct_change().dropna()
    n = r.shape[1]
    idx = r.index
    month_end = idx.to_series().groupby(idx.to_period("M")).transform("max")
    rebalance = pd.Series(idx == month_end.values, index=idx)

    w = np.ones(n, dtype=float) / n
    out: list[float] = []
    for i in range(len(idx)):
        row_ret = r.iloc[i].values.astype(float)
        pr = float(np.dot(w, row_ret))
        out.append(pr)
        drift = w * (1.0 + row_ret)
        v = float(drift.sum())
        w = drift / v if v > 1e-12 else np.ones(n) / n
        if bool(rebalance.iloc[i]):
            w = np.ones(n, dtype=float) / n
    return pd.Series(out, index=idx, name="equal_weight")


def markowitz_max_sharpe_monthly(
    prices: pd.DataFrame,
    lookback: int = 60,
    risk_free_annual: float = 0.02,
    freq: str = "ME",
) -> pd.Series:
    """
    Month-end tangency portfolio from past `lookback` daily returns; hold with drift.
    """
    del freq
    rf_d = _daily_rf(risk_free_annual)
    px = prices.astype(float).ffill().dropna(how="any")
    r = px.pct_change().dropna()
    n = r.shape[1]
    idx = r.index
    month_end = idx.to_series().groupby(idx.to_period("M")).transform("max")
    rebalance = pd.Series(idx == month_end.values, index=idx)

    w = np.ones(n, dtype=float) / n
    out: list[float] = []
    for i in range(len(idx)):
        if bool(rebalance.iloc[i]) and i >= lookback:
            hist = r.iloc[i - lookback : i].values
            mu = hist.mean(axis=0)
            cov = np.cov(hist, rowvar=False)
            cov = cov + np.eye(n) * 1e-8
            try:
                inv = np.linalg.inv(cov)
                ones = np.ones(n)
                mu_ex = mu - rf_d
                numer = inv @ mu_ex
                den = float(ones.T @ inv @ mu_ex)
                if abs(den) > 1e-12 and np.all(np.isfinite(numer)):
                    w_new = numer / den
                    w_new = np.maximum(w_new, 0.0)
                    s = w_new.sum()
                    if s > 1e-12:
                        w = w_new / s
                    else:
                        w = np.ones(n) / n
                else:
                    w = np.ones(n) / n
            except np.linalg.LinAlgError:
                w = np.ones(n) / n

        row_ret = r.iloc[i].values.astype(float)
        pr = float(np.dot(w, row_ret))
        out.append(pr)
        drift = w * (1.0 + row_ret)
        v = float(drift.sum())
        w = drift / v if v > 1e-12 else np.ones(n) / n

    return pd.Series(out, index=idx, name="markowitz")


def benchmark_suite(prices: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or {}
    rf = float(cfg.get("risk_free_annual", 0.02))
    lb = int(cfg.get("markowitz_lookback", 60))
    freq = cfg.get("equal_weight_freq", "ME")

    out = pd.DataFrame(
        {
            "buy_hold_eq": buy_and_hold_equal_initial(prices),
            "equal_weight": equal_weight_monthly(prices, freq=freq),
            "markowitz": markowitz_max_sharpe_monthly(
                prices, lookback=lb, risk_free_annual=rf, freq=freq
            ),
        }
    )
    return out.sort_index().dropna(how="all")
