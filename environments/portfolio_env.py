"""
Custom Gymnasium environment for multi-asset portfolio allocation with transaction costs.
"""

from __future__ import annotations

from typing import Any, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    z = x - np.max(x)
    e = np.exp(np.clip(z, -50, 50))
    return e / (np.sum(e) + 1e-12)


class PortfolioEnv(gym.Env):
    """
    State: flattened (lookback, n_assets, n_features) — per asset: return,
           rolling vol, RSI, normalized volume (standardized).

    Action: Box [0,1]^n_assets — softmax yields risky weights; mean(action)
            scales risky allocation vs cash.

    Reward: logarithmic return from portfolio value at step start to value after
            returns and rebalance (transaction costs applied).

    Transaction cost: proportional to turnover (50% of sum |Δw| over full
    weight vector including cash).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        lookback: int,
        transaction_cost: float = 0.001,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._rng = np.random.default_rng(seed)

        self.features = np.asarray(features, dtype=np.float32)
        self.returns = np.asarray(returns, dtype=np.float32)
        if self.features.ndim != 3:
            raise ValueError("features must have shape (T, n_assets, n_feat)")
        T, n_assets, n_feat = self.features.shape
        if self.returns.shape != (T, n_assets):
            raise ValueError("returns shape must match (T, n_assets)")
        if n_feat != 4:
            raise ValueError("Expected 4 features per asset")

        self.T = T
        self.n_assets = n_assets
        self.lookback = lookback
        self.transaction_cost = float(transaction_cost)

        flat_dim = lookback * n_assets * n_feat
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_assets,), dtype=np.float32
        )

        self._start: int = lookback
        self._t: int = lookback
        self._portfolio_value: float = 1.0
        self._w_risky: np.ndarray = np.ones(n_assets, dtype=np.float64) / n_assets
        self._w_cash: float = 0.0

    def action_to_target_weights(self, action: np.ndarray) -> Tuple[np.ndarray, float]:
        """Softmax on actions for risky split; mean(action) scales risky vs cash."""
        a = np.asarray(action, dtype=np.float64).flatten()
        risky_softmax = _softmax(a)
        rho = float(np.clip(np.mean(a), 0.0, 1.0))
        w_risky = risky_softmax * rho
        cash = float(max(0.0, 1.0 - np.sum(w_risky)))
        return w_risky, cash

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        opts = options or {}
        start = int(opts.get("start", self._start))
        start = max(self.lookback, min(start, self.T - 2))
        self._t = start
        self._portfolio_value = 1.0
        self._w_risky = np.ones(self.n_assets, dtype=np.float64) / self.n_assets
        self._w_cash = 0.0

        return self._observe(), {}

    def _observe(self) -> np.ndarray:
        lo = self._t - self.lookback
        hi = self._t
        window = self.features[lo:hi]
        return window.reshape(-1).astype(np.float32)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        r = self.returns[self._t].astype(np.float64)
        v_start = float(self._portfolio_value)

        w_risky_prev = self._w_risky.copy()
        w_cash_prev = self._w_cash

        values_stock = v_start * w_risky_prev * (1.0 + r)
        value_cash = v_start * w_cash_prev
        v_pre = float(np.sum(values_stock) + value_cash)
        if v_pre <= 1e-12:
            v_pre = 1e-12

        w_after = values_stock / v_pre
        w_cash_after = value_cash / v_pre

        w_target, cash_target = self.action_to_target_weights(action)
        w_full_before = np.append(w_after, w_cash_after)
        w_full_target = np.append(w_target, cash_target)

        turnover = 0.5 * float(np.sum(np.abs(w_full_target - w_full_before)))
        cost_frac = self.transaction_cost * turnover
        v_end = v_pre * (1.0 - cost_frac)

        self._w_risky = w_target.copy()
        self._w_cash = cash_target
        self._portfolio_value = float(v_end)

        reward = float(np.log(max(v_end / v_start, 1e-12)))

        self._t += 1
        terminated = self._t >= self.T - 1
        truncated = False
        info = {
            "portfolio_value": self._portfolio_value,
            "turnover": turnover,
            "cost_frac": cost_frac,
            "weights_risky": self._w_risky.copy(),
        }

        if terminated:
            obs = np.zeros_like(self.observation_space.sample())
        else:
            obs = self._observe()

        return obs, reward, terminated, truncated, info
