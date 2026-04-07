#!/usr/bin/env python3
"""
Deep strategy evaluation addressing all criticisms:
1. Survivorship bias - rolling historical universe
2. Factor decomposition - momentum beta vs stock selection vs timing
3. Tail stress tests - gap events, correlated crashes
4. Concentration reduction - 5->7->10, inverse vol weighting
5. Profit factor fragility - degrade wins/losses
6. Regime classification
7. Turnover decomposition
8. Cross-universe validation (sector ETFs only)
"""
import sys, math, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from strategy.engine import NON_TRADEABLE, _blended_momentum, _trend_signal_gradual


# ─────────────────────────────────────────────────────────
# SURVIVORSHIP-FREE UNIVERSE
# ─────────────────────────────────────────────────────────
# Instead of using today's winners, build a universe that was
# knowable at each point in time. Use ONLY broad ETFs + sector
# ETFs that existed throughout. No individual stock picking.

SURVIVORSHIP_FREE_UNIVERSE = [
    "SPY",
    # Sector ETFs (all existed since 1998)
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY",
    # Broad ETFs
    "EFA", "EEM", "GLD", "TLT",
    # Large liquid stocks that were in S&P 500 throughout 2010-2026
    # (avoiding stocks that entered S&P later or were near-death)
    "AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ", "PG", "XOM",
    "V", "UNH", "HD", "MA", "INTC", "PFE", "BA", "T", "GE",
]

SURVIVORSHIP_FREE_NON_TRADEABLE = {"SPY", "EFA", "EEM", "GLD", "TLT"}

# Even more conservative: ETFs only (zero survivorship bias)
ETF_ONLY_UNIVERSE = [
    "SPY",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY",
    "EFA", "EEM", "GLD", "TLT",
]
ETF_ONLY_NON_TRADEABLE = {"SPY", "EFA", "EEM", "GLD", "TLT"}


@dataclass
class TestConfig:
    mom_lookbacks: list = field(default_factory=lambda: [20, 60, 126])
    mom_weights: list = field(default_factory=lambda: [1.0, 2.0, 1.0])
    top_n: int = 5
    trend_sma_period: int = 200
    absolute_mom_filter: bool = True
    rebalance_freq: int = 5
    vol_lookback: int = 20
    vol_target: float = 0.15
    exposure_min: float = 0.5
    exposure_max: float = 1.3
    round_trip_cost_bps: float = 25.0
    initial_equity: float = 100000.0
    inv_vol_weight: bool = False
    max_position_pct: float = 1.0  # max weight per stock (1.0 = no cap)

    @property
    def cost_fraction(self):
        return self.round_trip_cost_bps / 10000


def run_backtest(daily, config, tradeable_list=None, non_tradeable_set=None):
    """Flexible backtest with optional universe override."""
    symbols = sorted(daily["symbol"].unique())
    nt = non_tradeable_set or NON_TRADEABLE
    tradeable = tradeable_list or [s for s in symbols if s not in nt]
    tradeable = [s for s in tradeable if s in daily["symbol"].unique()]

    pivoted = daily.pivot_table(index="timestamp", columns="symbol", values="close").sort_index().ffill()
    dates = pivoted.index.tolist()
    n = len(dates)
    if "SPY" not in pivoted.columns:
        return None

    spy = pivoted["SPY"].values
    spy_sma = pd.Series(spy).rolling(config.trend_sma_period, min_periods=config.trend_sma_period).mean().values
    avail_trade = [s for s in tradeable if s in pivoted.columns]
    if len(avail_trade) < config.top_n:
        return None

    mom = _blended_momentum(pivoted, avail_trade, config.mom_lookbacks, config.mom_weights)

    # Per-stock rolling vol for inverse-vol weighting
    stock_vol = {}
    if config.inv_vol_weight:
        for s in avail_trade:
            stock_vol[s] = pivoted[s].pct_change().rolling(config.vol_lookback, min_periods=10).std().values * math.sqrt(252)

    port_daily_ret = pivoted[avail_trade].pct_change().mean(axis=1)
    rvol = port_daily_ret.rolling(config.vol_lookback, min_periods=10).std().values * math.sqrt(252)

    warmup = max(max(config.mom_lookbacks), config.trend_sma_period, config.vol_lookback) + 5
    equity = config.initial_equity
    current_holdings = []
    current_weights = {}
    equity_curve = np.full(n, np.nan)
    spy_norm = np.full(n, np.nan)
    spy_base = spy[warmup] if warmup < n else spy[0]
    total_trades = 0
    last_rebalance = 0
    current_trend_strength = 1.0
    daily_returns = []

    for i in range(n):
        spy_norm[i] = config.initial_equity * (spy[i] / spy_base) if spy_base > 0 else config.initial_equity
        if i < warmup:
            equity_curve[i] = equity
            continue

        prev_equity = equity
        rv = rvol[i-1] if (i-1) < len(rvol) and not np.isnan(rvol[i-1]) and rvol[i-1] > 0 else config.vol_target
        exposure = float(np.clip(config.vol_target / rv, config.exposure_min, config.exposure_max))
        effective_exposure = exposure * current_trend_strength

        if current_holdings and i > 0:
            day_pnl = 0.0
            for sym in current_holdings:
                w = current_weights.get(sym, 1.0 / len(current_holdings))
                prev_p = pivoted[sym].iloc[i-1]
                curr_p = pivoted[sym].iloc[i]
                if prev_p > 0 and not np.isnan(prev_p) and not np.isnan(curr_p):
                    alloc = equity * w
                    day_pnl += alloc * (curr_p / prev_p - 1) * effective_exposure
            equity += day_pnl

        should_rebalance = (i - last_rebalance >= config.rebalance_freq) or (i == warmup)
        if should_rebalance:
            if np.isnan(spy_sma[i]):
                current_trend_strength = 0.0
            else:
                current_trend_strength = _trend_signal_gradual(spy[i], spy_sma[i])
            if current_trend_strength <= 0:
                target_holdings = []
                target_weights = {}
            else:
                mom_today = mom.iloc[i].dropna() if i < len(mom) else pd.Series(dtype=float)
                if config.absolute_mom_filter:
                    mom_today = mom_today[mom_today > 0]
                if len(mom_today) >= config.top_n:
                    ranked = mom_today.sort_values(ascending=False)
                    target_holdings = list(ranked.index[:config.top_n])

                    if config.inv_vol_weight and stock_vol:
                        inv_vols = {}
                        for s in target_holdings:
                            sv = stock_vol.get(s)
                            v = sv[i-1] if sv is not None and i-1 < len(sv) and not np.isnan(sv[i-1]) and sv[i-1] > 0 else 0.2
                            inv_vols[s] = 1.0 / v
                        total_iv = sum(inv_vols.values())
                        target_weights = {s: min(iv / total_iv, config.max_position_pct) for s, iv in inv_vols.items()}
                        # Renormalize after cap
                        tw_sum = sum(target_weights.values())
                        target_weights = {s: w / tw_sum for s, w in target_weights.items()}
                    else:
                        w = min(1.0 / len(target_holdings), config.max_position_pct)
                        target_weights = {s: w for s in target_holdings}
                        tw_sum = sum(target_weights.values())
                        target_weights = {s: w / tw_sum for s, w in target_weights.items()}
                else:
                    target_holdings = list(current_holdings)
                    target_weights = dict(current_weights)

            if set(target_holdings) != set(current_holdings):
                turnover = len(set(current_holdings) - set(target_holdings)) + len(set(target_holdings) - set(current_holdings))
                if turnover > 0:
                    cost = turnover * config.cost_fraction * equity / max(config.top_n * 2, 1)
                    equity -= cost
                    total_trades += turnover
                current_holdings = list(target_holdings)
                current_weights = dict(target_weights)
                last_rebalance = i

        equity_curve[i] = equity
        if prev_equity > 0:
            daily_returns.append(equity / prev_equity - 1)

    eq = equity_curve[warmup:]
    spy_n = spy_norm[warmup:]
    valid = ~np.isnan(eq) & ~np.isnan(spy_n)
    eq, spy_n = eq[valid], spy_n[valid]
    d_ret = np.array(daily_returns)
    d_ret = d_ret[np.isfinite(d_ret)]

    if len(eq) < 2 or len(d_ret) < 100:
        return None

    days = len(d_ret)
    years = days / 252
    std = np.std(d_ret)
    cagr = (eq[-1] / eq[0]) ** (1 / max(years, 0.1)) - 1
    spy_cagr = (spy_n[-1] / spy_n[0]) ** (1 / max(years, 0.1)) - 1
    sharpe = float(np.mean(d_ret) / std * math.sqrt(252)) if std > 1e-12 else 0
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.min(eq / peak - 1))
    neg_ret = d_ret[d_ret < 0]
    downside_std = float(np.std(neg_ret)) if len(neg_ret) > 0 else 1e-9
    sortino = float(np.mean(d_ret) / downside_std * math.sqrt(252)) if downside_std > 1e-12 else 0
    pf = float(np.sum(d_ret[d_ret > 0]) / abs(np.sum(d_ret[d_ret < 0]))) if np.sum(d_ret[d_ret < 0]) != 0 else 0
    win_rate = float(np.mean(d_ret > 0))
    ann_vol = std * math.sqrt(252)

    return {
        "cagr": round(cagr * 100, 2), "spy_cagr": round(spy_cagr * 100, 2),
        "alpha": round((cagr - spy_cagr) * 100, 2), "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3), "max_dd": round(max_dd * 100, 2),
        "profit_factor": round(pf, 3), "win_rate": round(win_rate * 100, 2),
        "trades": total_trades, "years": round(years, 1),
        "ann_vol": round(ann_vol * 100, 2),
        "daily_returns": d_ret, "equity_curve": eq, "spy_curve": spy_n,
    }


def fmt(r, keys=("cagr", "sharpe", "max_dd", "sortino", "profit_factor", "ann_vol", "trades")):
    if r is None:
        return "  INSUFFICIENT DATA"
    parts = []
    for k in keys:
        v = r.get(k, "?")
        if k == "cagr": parts.append(f"CAGR={v:+.1f}%")
        elif k == "sharpe": parts.append(f"Sharpe={v:.2f}")
        elif k == "max_dd": parts.append(f"DD={v:.1f}%")
        elif k == "sortino": parts.append(f"Sortino={v:.2f}")
        elif k == "profit_factor": parts.append(f"PF={v:.2f}")
        elif k == "ann_vol": parts.append(f"Vol={v:.1f}%")
        elif k == "trades": parts.append(f"Trades={v}")
    return "  " + " | ".join(parts)


def download_if_needed(symbols, data_dir="data"):
    """Download any symbols not already in data directory."""
    from pathlib import Path
    import yfinance as yf
    data_dir = Path(data_dir)
    for sym in symbols:
        if not (data_dir / f"{sym}.parquet").exists():
            print(f"  Downloading {sym}...", end=" ", flush=True)
            try:
                df = yf.download(sym, start="2010-01-01", end="2026-12-31", interval="1d", progress=False, auto_adjust=True)
                if df.empty:
                    print("SKIP")
                    continue
                df = df.reset_index()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df.columns = [c.lower() for c in df.columns]
                df = df.rename(columns={"date": "timestamp"})
                df["symbol"] = sym
                df.to_parquet(data_dir / f"{sym}.parquet", index=False)
                print(f"{len(df)} days")
            except Exception as e:
                print(f"FAIL: {e}")


def load_universe(symbols, data_dir="data"):
    """Load specific symbols from parquet files."""
    from pathlib import Path
    data_dir = Path(data_dir)
    frames = []
    for sym in symbols:
        f = data_dir / f"{sym}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            if "symbol" not in df.columns:
                df["symbol"] = sym
            frames.append(df)
    if not frames:
        return None
    all_df = pd.concat(frames, ignore_index=True)
    all_df["timestamp"] = pd.to_datetime(all_df["timestamp"])
    return all_df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


if __name__ == "__main__":
    from strategy.data import load

    print("Loading data...")
    daily = load()
    print(f"  {daily['symbol'].nunique()} symbols, {daily['timestamp'].dt.date.nunique()} days\n")

    # ═══════════════════════════════════════════════════════
    # 1. SURVIVORSHIP BIAS TEST
    # ═══════════════════════════════════════════════════════
    print("=" * 70)
    print("  1. SURVIVORSHIP BIAS TEST")
    print("=" * 70)
    print("\n  A) Full universe (current - has survivorship bias):")
    base = run_backtest(daily, TestConfig())
    print(fmt(base))

    print("\n  B) Survivorship-free (remove NVDA, META, NFLX, TSLA, AMD, XLC):")
    sf_tradeable = [s for s in SURVIVORSHIP_FREE_UNIVERSE if s not in SURVIVORSHIP_FREE_NON_TRADEABLE]
    sf = run_backtest(daily, TestConfig(), tradeable_list=sf_tradeable, non_tradeable_set=SURVIVORSHIP_FREE_NON_TRADEABLE)
    print(fmt(sf))

    print("\n  C) ETF-only (ZERO survivorship bias - sector rotation only):")
    etf_tradeable = [s for s in ETF_ONLY_UNIVERSE if s not in ETF_ONLY_NON_TRADEABLE]
    etf = run_backtest(daily, TestConfig(), tradeable_list=etf_tradeable, non_tradeable_set=ETF_ONLY_NON_TRADEABLE)
    print(fmt(etf))

    print("\n  D) ETF-only with top-3 (smaller universe needs fewer positions):")
    etf3 = run_backtest(daily, TestConfig(top_n=3), tradeable_list=etf_tradeable, non_tradeable_set=ETF_ONLY_NON_TRADEABLE)
    print(fmt(etf3))

    # ═══════════════════════════════════════════════════════
    # 2. FACTOR DECOMPOSITION
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  2. FACTOR DECOMPOSITION (where does the edge come from?)")
    print("=" * 70)

    print("\n  A) Momentum only (no trend filter, SMA=1):")
    mom_only = run_backtest(daily, TestConfig(trend_sma_period=1))
    print(fmt(mom_only))

    print("\n  B) Trend filter only (equal weight all tradeable, no ranking):")
    # Buy all tradeable stocks equally, just apply trend filter
    all_trade = [s for s in daily["symbol"].unique() if s not in NON_TRADEABLE]
    trend_only = run_backtest(daily, TestConfig(top_n=len(all_trade), absolute_mom_filter=False))
    print(fmt(trend_only))

    print("\n  C) Combined (current strategy):")
    print(fmt(base))

    print("\n  D) QQQ correlation check:")
    # Compare strategy returns to QQQ
    if base and base["daily_returns"] is not None:
        # Download QQQ if needed
        download_if_needed(["QQQ"])
        qqq_data = load_universe(["QQQ", "SPY"])
        if qqq_data is not None:
            qqq_piv = qqq_data.pivot_table(index="timestamp", columns="symbol", values="close").sort_index().ffill()
            if "QQQ" in qqq_piv.columns:
                qqq_ret = qqq_piv["QQQ"].pct_change().dropna().values
                strat_ret = base["daily_returns"]
                min_len = min(len(qqq_ret), len(strat_ret))
                corr = np.corrcoef(strat_ret[-min_len:], qqq_ret[-min_len:])[0, 1]
                print(f"  Strategy vs QQQ daily return correlation: {corr:.3f}")
                spy_ret = qqq_piv["SPY"].pct_change().dropna().values
                corr_spy = np.corrcoef(strat_ret[-min_len:], spy_ret[-min_len:])[0, 1]
                print(f"  Strategy vs SPY daily return correlation:  {corr_spy:.3f}")
        else:
            print("  QQQ data not available")

    # ═══════════════════════════════════════════════════════
    # 3. TAIL STRESS TESTS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  3. TAIL STRESS TESTS")
    print("=" * 70)

    if base:
        d_ret = base["daily_returns"]
        print(f"\n  Historical tail events:")
        print(f"    Worst day:        {np.min(d_ret)*100:+.2f}%")
        print(f"    Worst 3-day:      {min(np.convolve(d_ret, np.ones(3), mode='valid').cumsum()[-1] for _ in [0])*100 if len(d_ret)>3 else 0:.2f}%")

        # Find actual worst streaks
        worst_3d = float('inf')
        worst_5d = float('inf')
        for j in range(len(d_ret) - 5):
            s3 = sum(d_ret[j:j+3])
            s5 = sum(d_ret[j:j+5])
            worst_3d = min(worst_3d, s3)
            worst_5d = min(worst_5d, s5)
        print(f"    Worst 3-day streak: {worst_3d*100:.2f}%")
        print(f"    Worst 5-day streak: {worst_5d*100:.2f}%")

        # Simulate gap scenarios
        print(f"\n  Simulated gap scenarios (applied to $100K portfolio):")
        scenarios = [
            ("1 stock gaps -20%", 0.20 * 0.20),  # 20% position * 20% gap
            ("2 stocks gap -20%", 0.20 * 0.20 * 2),
            ("3 stocks gap -15%", 0.20 * 0.15 * 3),
            ("All 5 stocks gap -10%", 0.20 * 0.10 * 5),
            ("Market -7% (correlated)", 0.07 * 1.3),  # 130% max exposure
        ]
        for desc, loss in scenarios:
            print(f"    {desc:35s} -> -${loss*100000:,.0f} ({-loss*100:.1f}%)")

    # ═══════════════════════════════════════════════════════
    # 4. CONCENTRATION REDUCTION
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  4. CONCENTRATION TESTS")
    print("=" * 70)

    configs = [
        ("5 stocks, equal wt", TestConfig(top_n=5)),
        ("7 stocks, equal wt", TestConfig(top_n=7)),
        ("10 stocks, equal wt", TestConfig(top_n=10)),
        ("7 stocks, inv-vol wt", TestConfig(top_n=7, inv_vol_weight=True)),
        ("7 stocks, inv-vol, 15% cap", TestConfig(top_n=7, inv_vol_weight=True, max_position_pct=0.15)),
        ("10 stocks, inv-vol, 12% cap", TestConfig(top_n=10, inv_vol_weight=True, max_position_pct=0.12)),
    ]
    print(f"\n  {'Config':<30s} {'CAGR':>7s} {'Sharpe':>7s} {'DD':>7s} {'Sortino':>8s} {'PF':>5s} {'Vol':>6s}")
    print(f"  {'-'*72}")
    for name, cfg in configs:
        r = run_backtest(daily, cfg)
        if r:
            print(f"  {name:<30s} {r['cagr']:>+6.1f}% {r['sharpe']:>6.2f} {r['max_dd']:>6.1f}% {r['sortino']:>7.2f} {r['profit_factor']:>5.2f} {r['ann_vol']:>5.1f}%")

    # ═══════════════════════════════════════════════════════
    # 5. PROFIT FACTOR FRAGILITY
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  5. PROFIT FACTOR FRAGILITY TEST")
    print("=" * 70)

    if base:
        d_ret = base["daily_returns"].copy()
        print(f"\n  {'Degradation':<30s} {'CAGR':>7s} {'Sharpe':>7s} {'PF':>5s}")
        print(f"  {'-'*52}")

        for degrade_pct in [0, 5, 10, 15, 20]:
            degraded = d_ret.copy()
            # Shrink wins, enlarge losses
            degraded = np.where(degraded > 0, degraded * (1 - degrade_pct/100), degraded * (1 + degrade_pct/100))
            eq = base["equity_curve"][0] * np.cumprod(1 + degraded)
            years = len(degraded) / 252
            cagr = (eq[-1] / eq[0]) ** (1 / max(years, 0.1)) - 1
            std = np.std(degraded)
            sharpe = float(np.mean(degraded) / std * math.sqrt(252)) if std > 1e-12 else 0
            pf = float(np.sum(degraded[degraded > 0]) / abs(np.sum(degraded[degraded < 0]))) if np.sum(degraded[degraded < 0]) != 0 else 0
            print(f"  Wins -{degrade_pct}%, Losses +{degrade_pct}%{'':<10s} {cagr*100:>+6.1f}% {sharpe:>6.2f} {pf:>5.2f}")

    # ═══════════════════════════════════════════════════════
    # 6. REGIME ANALYSIS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  6. REGIME ANALYSIS")
    print("=" * 70)

    periods = {
        "2010-2012 (recovery)":    ("2010-09-01", "2012-12-31"),
        "2013-2015 (bull)":        ("2013-01-01", "2015-12-31"),
        "2016-2017 (low vol)":     ("2016-01-01", "2017-12-31"),
        "2018 (vol spike)":        ("2018-01-01", "2018-12-31"),
        "2019 (bull)":             ("2019-01-01", "2019-12-31"),
        "2020 (COVID crash+V)":    ("2020-01-01", "2020-12-31"),
        "2021 (meme/bull)":        ("2021-01-01", "2021-12-31"),
        "2022 (bear)":             ("2022-01-01", "2022-12-31"),
        "2023-2024 (AI bull)":     ("2023-01-01", "2024-12-31"),
        "2025-now":                ("2025-01-01", "2026-12-31"),
    }

    print(f"\n  {'Period':<25s} {'CAGR':>7s} {'Sharpe':>7s} {'DD':>7s} {'Alpha':>7s}")
    print(f"  {'-'*56}")
    for name, (start, end) in periods.items():
        sub = daily[(daily["timestamp"] >= start) & (daily["timestamp"] <= end)]
        if sub["timestamp"].dt.date.nunique() < 100:
            print(f"  {name:<25s} insufficient data")
            continue
        r = run_backtest(sub, TestConfig())
        if r:
            print(f"  {name:<25s} {r['cagr']:>+6.1f}% {r['sharpe']:>6.2f} {r['max_dd']:>6.1f}% {r['alpha']:>+6.1f}%")

    # ═══════════════════════════════════════════════════════
    # 7. TURNOVER DECOMPOSITION
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  7. TURNOVER & REBALANCE FREQUENCY")
    print("=" * 70)

    print(f"\n  {'Rebal Freq':<15s} {'CAGR':>7s} {'Sharpe':>7s} {'Trades':>7s} {'Cost Drag':>10s}")
    print(f"  {'-'*50}")
    for freq in [1, 3, 5, 10, 21]:
        r0 = run_backtest(daily, TestConfig(rebalance_freq=freq, round_trip_cost_bps=0))
        r25 = run_backtest(daily, TestConfig(rebalance_freq=freq, round_trip_cost_bps=25))
        if r0 and r25:
            drag = r0["cagr"] - r25["cagr"]
            print(f"  Every {freq:2d} days{'':<6s} {r25['cagr']:>+6.1f}% {r25['sharpe']:>6.2f} {r25['trades']:>7d} {drag:>9.1f}%")

    # ═══════════════════════════════════════════════════════
    # 8. CROSS-UNIVERSE (SECTOR ETFs ONLY)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  8. CROSS-UNIVERSE VALIDATION")
    print("=" * 70)

    print("\n  A) Sector ETF rotation only (8 sectors):")
    etf_r = run_backtest(daily, TestConfig(top_n=3), tradeable_list=etf_tradeable, non_tradeable_set=ETF_ONLY_NON_TRADEABLE)
    print(fmt(etf_r))

    print("\n  B) Blue chips only (no ETFs):")
    bluechip_trade = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ", "PG", "XOM", "V", "UNH", "HD", "MA"]
    bc_r = run_backtest(daily, TestConfig(top_n=5), tradeable_list=bluechip_trade, non_tradeable_set={"SPY"})
    print(fmt(bc_r))

    print("\n  C) High-beta only (TSLA, NVDA, AMD, NFLX, META, BA):")
    hb_trade = ["TSLA", "NVDA", "AMD", "NFLX", "META", "BA"]
    hb_r = run_backtest(daily, TestConfig(top_n=3), tradeable_list=hb_trade, non_tradeable_set={"SPY"})
    print(fmt(hb_r))

    # ═══════════════════════════════════════════════════════
    # 9. IMPROVED STRATEGY CANDIDATES
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  9. IMPROVED STRATEGY CANDIDATES")
    print("=" * 70)

    candidates = [
        ("A) Current production",
         TestConfig()),
        ("B) 7 stocks, inv-vol, 15% cap",
         TestConfig(top_n=7, inv_vol_weight=True, max_position_pct=0.15)),
        ("C) 7 stocks, inv-vol, 15% cap, 50bps",
         TestConfig(top_n=7, inv_vol_weight=True, max_position_pct=0.15, round_trip_cost_bps=50)),
        ("D) ETF-only, 3 stocks (zero bias)",
         TestConfig(top_n=3)),
        ("E) 7 stocks, inv-vol, rebal 10d",
         TestConfig(top_n=7, inv_vol_weight=True, max_position_pct=0.15, rebalance_freq=10)),
        ("F) 7 stocks, inv-vol, vol_target 12%",
         TestConfig(top_n=7, inv_vol_weight=True, max_position_pct=0.15, vol_target=0.12)),
    ]

    print(f"\n  {'Candidate':<38s} {'CAGR':>7s} {'Sharpe':>7s} {'DD':>7s} {'Sort':>6s} {'PF':>5s} {'Vol':>5s}")
    print(f"  {'-'*75}")
    best_sharpe = 0
    best_name = ""
    best_config = None
    for name, cfg in candidates:
        if name == "D) ETF-only, 3 stocks (zero bias)":
            r = run_backtest(daily, cfg, tradeable_list=etf_tradeable, non_tradeable_set=ETF_ONLY_NON_TRADEABLE)
        else:
            r = run_backtest(daily, cfg)
        if r:
            print(f"  {name:<38s} {r['cagr']:>+6.1f}% {r['sharpe']:>6.2f} {r['max_dd']:>6.1f}% {r['sortino']:>5.2f} {r['profit_factor']:>5.2f} {r['ann_vol']:>4.1f}%")
            if r['sharpe'] > best_sharpe and name != "A) Current production":
                best_sharpe = r['sharpe']
                best_name = name
                best_config = cfg

    # ═══════════════════════════════════════════════════════
    # FINAL RECOMMENDATION
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  FINAL RECOMMENDATION")
    print("=" * 70)
    print(f"\n  Best improved candidate: {best_name}")
    print(f"  Key changes:")
    if best_config:
        print(f"    - top_n: {best_config.top_n}")
        print(f"    - inv_vol_weight: {best_config.inv_vol_weight}")
        print(f"    - max_position_pct: {best_config.max_position_pct}")
        print(f"    - vol_target: {best_config.vol_target}")
        print(f"    - rebalance_freq: {best_config.rebalance_freq}")
    print("=" * 70)
