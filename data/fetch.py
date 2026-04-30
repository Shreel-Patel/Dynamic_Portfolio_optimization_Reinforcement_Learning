"""Download adjusted daily prices via yfinance and save raw CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: Path | None = None) -> dict:
    root = Path(__file__).resolve().parents[1]
    cfg_path = path or root / "config" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    out_dir: Path,
) -> pd.DataFrame:
    import yfinance as yf

    out_dir.mkdir(parents=True, exist_ok=True)

    data = yf.download(
        " ".join(tickers),
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        group_by="column",
        threads=True,
    )
    if data.empty:
        raise RuntimeError("No price data returned")

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data[["Close"]].copy()
        close.columns = tickers[: close.shape[1]]

    close = close[tickers].astype(float)
    close = close.sort_index().ffill().dropna(how="any")
    out_path = out_dir / "adj_close.csv"
    close.to_csv(out_path)
    print(f"Saved {out_path} shape={close.shape}")
    return close


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch yfinance daily adjusted closes")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tickers = cfg["tickers"]
    start = cfg["data"]["start"]
    end = cfg["data"]["end"]
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    fetch_prices(tickers, start, end, raw_dir)


if __name__ == "__main__":
    main()
