#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/calle/Trader')
from strategy.data import fetch_live
df = fetch_live()
spy = df[df['symbol'] == 'SPY']
print(f"Total rows: {len(df)}")
print(f"SPY rows: {len(spy)}")
print(f"SPY date range: {spy['timestamp'].min()} to {spy['timestamp'].max()}")
print(f"Unique dates: {df['timestamp'].nunique()}")

import pandas as pd
pivoted = df.pivot_table(index='timestamp', columns='symbol', values='close').sort_index().ffill()
spy_s = pivoted['SPY']
spy_sma = spy_s.rolling(100, min_periods=100).mean()
print(f"\nSPY series length: {len(spy_s)}")
print(f"SMA non-null: {spy_sma.notna().sum()}")
print(f"SMA null: {spy_sma.isna().sum()}")

tail = spy_s.iloc[-200:]
tail_sma = spy_sma.iloc[-200:]
nulls = tail_sma.isna().sum()
print(f"\nLast 200 SPY points: {len(tail)}")
print(f"Last 200 SMA nulls: {nulls}")
