"""Download daily OHLCV from Yahoo Finance."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from strategy.engine import UNIVERSE


def download(data_dir: str | Path = "data") -> Path:
    """Download daily bars for all universe symbols. Returns data directory."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for sym in UNIVERSE:
        print(f"  {sym}...", end=" ", flush=True)
        try:
            df = yf.download(
                sym, start="2010-01-01", end="2026-12-31",
                interval="1d", progress=False, auto_adjust=True,
            )
        except Exception as e:
            print(f"FAIL ({e})")
            continue
        if df.empty:
            print("SKIP")
            continue

        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"date": "timestamp"})
        df["symbol"] = sym

        out = data_dir / f"{sym}.parquet"
        df.to_parquet(out, index=False)
        print(f"{len(df)} days ({df['timestamp'].min().date()} to {df['timestamp'].max().date()})")

    return data_dir


def load(data_dir: str | Path = "data") -> pd.DataFrame:
    """Load all downloaded parquet files into a single DataFrame."""
    data_dir = Path(data_dir)
    frames = []
    for f in sorted(data_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        if "symbol" not in df.columns:
            df["symbol"] = f.stem
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No parquet files in {data_dir}. Run: python run.py download")
    all_df = pd.concat(frames, ignore_index=True)
    all_df["timestamp"] = pd.to_datetime(all_df["timestamp"])
    return all_df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def fetch_live(symbols: list[str] | None = None, lookback_days: int = 600) -> pd.DataFrame:
    """Fetch recent daily prices from Yahoo Finance for live signal computation."""
    from datetime import datetime, timedelta

    symbols = symbols or UNIVERSE
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    frames = []

    for sym in symbols:
        try:
            df = yf.download(
                sym, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                interval="1d", progress=False, auto_adjust=True,
            )
            if df.empty:
                continue
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={"date": "timestamp"})
            df["symbol"] = sym
            frames.append(df)
        except Exception:
            continue

    if not frames:
        raise RuntimeError("Could not fetch any price data")
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"])


if __name__ == "__main__":
    download()
