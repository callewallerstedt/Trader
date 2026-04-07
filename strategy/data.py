"""Download and manage daily OHLCV from Yahoo Finance."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from strategy.engine import UNIVERSE

log = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2


def _download_symbol(sym: str, start: str, end: str) -> pd.DataFrame | None:
    """Download a single symbol with retry logic."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(
                sym, start=start, end=end,
                interval="1d", progress=False, auto_adjust=True,
            )
            if df.empty:
                return None
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={"date": "timestamp"})
            df["symbol"] = sym
            return df
        except Exception as e:
            if attempt < MAX_RETRIES:
                log.warning(f"{sym}: attempt {attempt} failed ({e}), retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY * attempt)
            else:
                log.error(f"{sym}: all {MAX_RETRIES} attempts failed: {e}")
                return None


def download(data_dir: str | Path = "data") -> Path:
    """Download daily bars for all universe symbols. Returns data directory."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    for sym in UNIVERSE:
        print(f"  {sym}...", end=" ", flush=True)
        df = _download_symbol(sym, "2010-01-01", "2026-12-31")
        if df is None:
            print("SKIP")
            continue

        out = data_dir / f"{sym}.parquet"
        df.to_parquet(out, index=False)
        print(f"{len(df)} days ({df['timestamp'].min().date()} to {df['timestamp'].max().date()})")
        success += 1

    log.info(f"Download complete: {success}/{len(UNIVERSE)} symbols")
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
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    frames = []
    failed = []

    for sym in symbols:
        df = _download_symbol(sym, start_str, end_str)
        if df is not None:
            frames.append(df)
        else:
            failed.append(sym)

    if failed:
        log.warning(f"Failed to fetch: {', '.join(failed)}")

    if not frames:
        raise RuntimeError("Could not fetch any price data")

    if "SPY" not in [f["symbol"].iloc[0] for f in frames]:
        raise RuntimeError("SPY data missing - cannot compute trend filter")

    result = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"])
    log.info(f"Fetched {result['symbol'].nunique()} symbols, "
             f"{result['timestamp'].dt.date.nunique()} days")
    return result


if __name__ == "__main__":
    download()
