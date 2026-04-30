"""
Feature engineering and train/val/test splits without look-ahead bias.
Rolling z-score uses past window only; standardized series shifted by one bar where needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import ta
except ImportError:
    ta = None


def load_config(path: Path | None = None) -> dict:
    root = Path(__file__).resolve().parents[1]
    cfg_path = path or root / "config" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def rolling_zscore(s: pd.Series, window: int = 252, min_periods: int = 20) -> pd.Series:
    """Z-score using past-only statistics (shift applied so current bar uses history)."""
    mu = s.rolling(window=window, min_periods=min_periods).mean().shift(1)
    sig = s.rolling(window=window, min_periods=min_periods).std().shift(1)
    z = (s - mu) / (sig + 1e-12)
    return z


def build_features_for_ticker(close: pd.Series, volume: pd.Series | None) -> pd.DataFrame:
    """Returns DataFrame with raw_ret, vol20, rsi14, vol_norm per ticker."""
    ret = close.pct_change()
    vol20 = ret.rolling(20).std()
    if ta is not None:
        rsi14 = ta.momentum.RSIIndicator(close, window=14).rsi()
    else:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        rsi14 = 100 - (100 / (1 + rs))

    if volume is not None and not volume.empty:
        vma = volume.rolling(20).mean()
        vol_norm = volume / (vma + 1e-12)
    else:
        vol_norm = pd.Series(1.0, index=close.index)

    out = pd.DataFrame(
        {
            "raw_ret": ret,
            "vol20": vol20,
            "rsi14": rsi14 / 100.0,
            "vol_norm": vol_norm,
        },
        index=close.index,
    )
    return out


def preprocess(
    raw_csv: Path,
    cfg: dict,
    processed_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    tickers = cfg["tickers"]
    train_end = pd.Timestamp(cfg["data"]["train_end"])
    val_start = pd.Timestamp(cfg["data"]["val_start"])
    val_end = pd.Timestamp(cfg["data"]["val_end"])
    test_start = pd.Timestamp(cfg["data"]["test_start"])

    prices = pd.read_csv(raw_csv, index_col=0, parse_dates=True).sort_index()

    feat_blocks = []
    for t in tickers:
        if t not in prices.columns:
            raise KeyError(f"Ticker {t} missing in raw data")
        px = prices[t].astype(float)
        raw = build_features_for_ticker(px, None)

        zcols = {}
        for col in ["raw_ret", "vol20", "rsi14", "vol_norm"]:
            zcols[col] = rolling_zscore(raw[col].astype(float))
        zdf = pd.DataFrame(zcols, index=raw.index)
        zdf.columns = [f"{t}_{c}" for c in zdf.columns]
        feat_blocks.append(zdf)

    wide = pd.concat(feat_blocks, axis=1).sort_index()

    rets = prices.pct_change()
    rets.columns = [f"{c}_ret_next" for c in rets.columns]

    merged = wide.join(rets, how="inner").dropna()

    meta = {
        "train_end": str(train_end.date()),
        "val_start": str(val_start.date()),
        "val_end": str(val_end.date()),
        "test_start": str(test_start.date()),
    }

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_csv = processed_dir / "features_returns.csv"
    merged.to_csv(out_csv)
    meta_path = processed_dir / "split_meta.yaml"
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.dump(meta, f)

    print(f"Saved {out_csv} shape={merged.shape}")
    return merged, meta


def dataframe_to_tensors(
    df: pd.DataFrame,
    tickers: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build (T, n_assets, 4) features and (T, n_assets) returns."""
    feat_cols = ["raw_ret", "vol20", "rsi14", "vol_norm"]
    T = len(df)
    n = len(tickers)
    feat = np.zeros((T, n, 4), dtype=np.float32)
    ret = np.zeros((T, n), dtype=np.float32)
    for i, t in enumerate(tickers):
        for k, c in enumerate(feat_cols):
            feat[:, i, k] = df[f"{t}_{c}"].values.astype(np.float32)
        ret[:, i] = df[f"{t}_ret_next"].values.astype(np.float32)
    return feat, ret


def split_indices(dates: pd.DatetimeIndex, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = pd.Timestamp(cfg["data"]["train_end"])
    val_start = pd.Timestamp(cfg["data"]["val_start"])
    val_end = pd.Timestamp(cfg["data"]["val_end"])
    test_start = pd.Timestamp(cfg["data"]["test_start"])

    train_mask = dates <= train_end
    val_mask = (dates >= val_start) & (dates <= val_end)
    test_mask = dates >= test_start

    return np.where(train_mask)[0], np.where(val_mask)[0], np.where(test_mask)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(__file__).resolve().parents[1]
    raw_csv = root / "data" / "raw" / "adj_close.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(f"Run fetch first: missing {raw_csv}")
    preprocess(raw_csv, cfg, root / "data" / "processed")


if __name__ == "__main__":
    main()
