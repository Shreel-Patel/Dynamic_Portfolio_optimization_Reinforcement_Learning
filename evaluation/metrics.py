"""Performance metrics for portfolio return series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna().astype(float)
    if r.empty:
        return float("nan")
    years = len(r) / 252.0
    cum = float(np.prod(1.0 + r.values) - 1.0)
    return float((1.0 + cum) ** (1.0 / max(years, 1e-9)) - 1.0)


def annualized_volatility(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna().astype(float)
    if len(r) < 2:
        return float("nan")
    return float(r.std() * np.sqrt(252.0))


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_annual: float = 0.02,
) -> float:
    rf_d = (1.0 + risk_free_annual) ** (1.0 / 252.0) - 1.0
    excess = daily_returns.dropna().astype(float) - rf_d
    if excess.std() < 1e-12:
        return float("nan")
    return float(excess.mean() / excess.std() * np.sqrt(252.0))


def max_drawdown(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna().astype(float)
    if r.empty:
        return float("nan")
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = (wealth / peak) - 1.0
    return float(dd.min())


def calmar_ratio(daily_returns: pd.Series) -> float:
    ann_ret = annualized_return(daily_returns)
    mdd = abs(max_drawdown(daily_returns))
    if mdd < 1e-12:
        return float("nan")
    return float(ann_ret / mdd)


def turnover(weights: pd.DataFrame) -> float:
    """Average daily turnover: mean sum(|Δw|) / 2 aligned with plan-style turnover."""
    if weights.shape[0] < 2:
        return float("nan")
    dw = weights.diff().abs().sum(axis=1).iloc[1:]
    return float(dw.mean())


def summarize_returns(
    daily_returns: pd.Series,
    risk_free_annual: float = 0.02,
    transaction_cost_total: float | None = None,
) -> dict[str, float]:
    out = {
        "annualized_return": annualized_return(daily_returns),
        "annualized_vol": annualized_volatility(daily_returns),
        "sharpe": sharpe_ratio(daily_returns, risk_free_annual),
        "max_drawdown": max_drawdown(daily_returns),
        "calmar": calmar_ratio(daily_returns),
    }
    if transaction_cost_total is not None:
        out["transaction_cost_total"] = float(transaction_cost_total)
    return out


def metrics_table(series_dict: dict[str, pd.Series], risk_free_annual: float = 0.02) -> pd.DataFrame:
    rows = []
    for name, s in series_dict.items():
        m = summarize_returns(s, risk_free_annual=risk_free_annual)
        m["strategy"] = name
        rows.append(m)
    return pd.DataFrame(rows).set_index("strategy")
