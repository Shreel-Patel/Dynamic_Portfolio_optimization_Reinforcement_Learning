"""Train SAC, DDPG, PPO with validation Sharpe checkpointing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import yaml
from stable_baselines3 import DDPG, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.preprocess import dataframe_to_tensors, split_indices  # noqa: E402
from environments.portfolio_env import PortfolioEnv  # noqa: E402


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or ROOT / "config" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _daily_rf(risk_free_annual: float = 0.02) -> float:
    return (1.0 + risk_free_annual) ** (1.0 / 252.0) - 1.0


def rollout_validation_sharpe(
    model,
    vec_env,
    risk_free_annual: float = 0.02,
) -> float:
    """One full episode on validation env; Sharpe on daily simple returns."""
    rf_d = _daily_rf(risk_free_annual)
    reset_out = vec_env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    rewards_log: list[float] = []
    for _ in range(500000):
        action, _ = model.predict(obs, deterministic=True)
        step_out = vec_env.step(action)
        if len(step_out) == 5:
            obs, rew, done, _, infos = step_out
        else:
            obs, rew, done, infos = step_out
        rewards_log.append(float(np.asarray(rew).reshape(-1)[0]))
        if np.asarray(done).reshape(-1)[0]:
            break
    log_r = np.asarray(rewards_log, dtype=float)
    if log_r.size < 10:
        return float("-inf")
    daily_simple = np.expm1(log_r)
    xs = daily_simple - rf_d
    if float(np.std(xs)) < 1e-12:
        return float("-inf")
    return float(np.mean(xs) / np.std(xs) * np.sqrt(252.0))


class SharpeCheckpointCallback(BaseCallback):
    """Periodically evaluate Sharpe on validation vec env and save best checkpoint."""

    def __init__(
        self,
        eval_vec_env,
        eval_freq: int,
        save_path: Path,
        risk_free_annual: float = 0.02,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_vec_env = eval_vec_env
        self.eval_freq = eval_freq
        self.save_path = Path(save_path)
        self.risk_free_annual = risk_free_annual
        self.best_sharpe = float("-inf")

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            sharpe = rollout_validation_sharpe(
                self.model, self.eval_vec_env, self.risk_free_annual
            )
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.save_path))
                if self.verbose:
                    print(f"[SharpeCheckpoint] new best Sharpe={sharpe:.4f} -> {self.save_path}")
        return True


def make_vec_env_fn(
    features: np.ndarray,
    returns: np.ndarray,
    lookback: int,
    transaction_cost: float,
    seed: int,
) -> Callable[[], PortfolioEnv]:
    def _thunk() -> PortfolioEnv:
        env = PortfolioEnv(
            features,
            returns,
            lookback,
            transaction_cost=transaction_cost,
            seed=seed,
        )
        return Monitor(env)

    return _thunk


def create_agent(name: str, env, cfg: dict, tensorboard_log: str | None):
    name_u = name.upper()
    common = dict(verbose=0, tensorboard_log=tensorboard_log, seed=cfg["training"]["seed"])
    if name_u == "SAC":
        p = cfg["agents"]["SAC"]
        return SAC(
            p["policy"],
            env,
            learning_rate=p["learning_rate"],
            buffer_size=p["buffer_size"],
            gamma=p["gamma"],
            batch_size=p["batch_size"],
            **common,
        )
    if name_u == "DDPG":
        p = cfg["agents"]["DDPG"]
        return DDPG(
            p["policy"],
            env,
            learning_rate=p["learning_rate"],
            buffer_size=p["buffer_size"],
            gamma=p["gamma"],
            batch_size=p["batch_size"],
            **common,
        )
    if name_u == "PPO":
        p = cfg["agents"]["PPO"]
        return PPO(
            p["policy"],
            env,
            learning_rate=p["learning_rate"],
            gamma=p["gamma"],
            batch_size=p["batch_size"],
            n_steps=2048,
            **common,
        )
    raise ValueError(f"Unknown agent {name}")


def train_all(
    cfg: dict | None = None,
    agents: list[str] | None = None,
    total_timesteps: int | None = None,
) -> None:
    cfg = cfg or load_config()
    agents = agents or ["SAC", "DDPG", "PPO"]

    np.random.seed(cfg["training"]["seed"])
    try:
        import torch

        torch.manual_seed(cfg["training"]["seed"])
    except Exception:
        pass

    data_csv = ROOT / "data" / "processed" / "features_returns.csv"
    if not data_csv.exists():
        raise FileNotFoundError(f"Missing processed data: {data_csv}")

    df = __import__("pandas").read_csv(data_csv, index_col=0, parse_dates=True)
    tickers = cfg["tickers"]
    feat_full, ret_full = dataframe_to_tensors(df, tickers)

    train_idx, val_idx, _test_idx = split_indices(df.index, cfg)
    if train_idx.size < 50 or val_idx.size < 20:
        raise RuntimeError("Insufficient rows for train/validation")

    lookback = int(cfg["environment"]["lookback"])
    tc = float(cfg["environment"]["transaction_cost"])

    feat_train = feat_full[train_idx]
    ret_train = ret_full[train_idx]
    feat_val = feat_full[val_idx]
    ret_val = ret_full[val_idx]

    tb_root = ROOT / "results" / "logs"
    models_dir = ROOT / "results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_fn = make_vec_env_fn(
        feat_train, ret_train, lookback, tc, cfg["training"]["seed"]
    )
    train_vec = DummyVecEnv([train_fn])

    val_fn = make_vec_env_fn(feat_val, ret_val, lookback, tc, cfg["training"]["seed"] + 1)
    val_vec = DummyVecEnv([val_fn])

    total_ts = int(total_timesteps or cfg["training"]["total_timesteps"])
    eval_freq = int(cfg["training"]["eval_freq"])
    if eval_freq > total_ts:
        eval_freq = max(total_ts // 5, 1)

    for name in agents:
        print(f"=== Training {name} ===")
        model = create_agent(
            name,
            train_vec,
            cfg,
            tensorboard_log=str(tb_root / name),
        )
        best_path = models_dir / f"{name.upper()}_best"
        cb = SharpeCheckpointCallback(
            val_vec,
            eval_freq=eval_freq,
            save_path=best_path,
            risk_free_annual=float(cfg["environment"]["risk_free_annual"]),
            verbose=1,
        )
        model.learn(total_timesteps=total_ts, callback=cb, progress_bar=True)

        final_path = models_dir / f"{name.upper()}_final"
        model.save(str(final_path))
        print(f"Saved final: {final_path}.zip")

        if cb.best_sharpe > float("-inf"):
            print(f"Best validation Sharpe ({name}): {cb.best_sharpe:.4f}")
        else:
            print(f"Warning: no Sharpe improvement tracked for {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--agents",
        nargs="*",
        default=None,
        help="Subset of SAC DDPG PPO (default: all)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override config training.total_timesteps (e.g. quick debug)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_all(cfg, agents=args.agents, total_timesteps=args.timesteps)


if __name__ == "__main__":
    main()
