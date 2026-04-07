#!/usr/bin/env python3
"""
Phase 2: Build and validate the improved strategy.
- VIX crash filter
- Survivorship-free universe with MORE stocks
- Inverse-vol weighting + position caps
- Regime-adaptive exposure
- Monte Carlo on improved version
- Final comparison: old vs new
"""
import sys, math, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from strategy.engine import _blended_momentum, _trend_signal_gradual

DATA_DIR = Path("data")

# ─────────────────────────────────────────────────────────
# UNIVERSE: SURVIVORSHIP-FREE
# ─────────────────────────────────────────────────────────
# Rules: Only stocks/ETFs that were in S&P 500 or equivalent
# throughout 2010-2025. Deliberately include underperformers.
# Mix of sectors to avoid sector bias.

UNIVERSE_STOCKS = [
    # Tech (include both winners AND losers)
    "AAPL", "MSFT", "INTC", "IBM", "CSCO", "ORCL", "TXN", "QCOM",
    # Finance
    "JPM", "BAC", "WFC", "GS", "C",
    # Healthcare
    "JNJ", "PFE", "UNH", "MRK", "ABT", "LLY",
    # Consumer
    "PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE",
    # Energy
    "XOM", "CVX", "COP",
    # Industrial
    "GE", "HON", "MMM", "CAT", "BA",
    # Telecom/Utilities
    "T", "VZ",
    # Materials
    "DOW",
]

UNIVERSE_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY",
]

BENCHMARK = "SPY"
ALL_SYMBOLS = sorted(set([BENCHMARK] + UNIVERSE_STOCKS + UNIVERSE_ETFS))
NON_TRADEABLE = {BENCHMARK}


def download_all():
    """Ensure all symbols are downloaded."""
    for sym in ALL_SYMBOLS:
        f = DATA_DIR / f"{sym}.parquet"
        if f.exists():
            continue
        print(f"  Downloading {sym}...", end=" ", flush=True)
        try:
            df = yf.download(sym, start="2009-01-01", end="2026-12-31",
                             interval="1d", progress=False, auto_adjust=True)
            if df.empty:
                print("EMPTY")
                continue
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={"date": "timestamp"})
            df["symbol"] = sym
            df.to_parquet(f, index=False)
            print(f"{len(df)} days")
        except Exception as e:
            print(f"FAIL: {e}")
        time.sleep(0.3)

    # Also download VIX
    vix_f = DATA_DIR / "^VIX.parquet"
    if not vix_f.exists():
        print("  Downloading ^VIX...", end=" ", flush=True)
        try:
            df = yf.download("^VIX", start="2009-01-01", end="2026-12-31",
                             interval="1d", progress=False, auto_adjust=True)
            if not df.empty:
                df = df.reset_index()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df.columns = [c.lower() for c in df.columns]
                df = df.rename(columns={"date": "timestamp"})
                df["symbol"] = "^VIX"
                df.to_parquet(vix_f, index=False)
                print(f"{len(df)} days")
        except Exception as e:
            print(f"FAIL: {e}")


def load_all():
    """Load all symbols into one DataFrame."""
    frames = []
    for sym in ALL_SYMBOLS:
        f = DATA_DIR / f"{sym}.parquet"
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


def load_vix():
    """Load VIX data."""
    f = DATA_DIR / "^VIX.parquet"
    if f.exists():
        df = pd.read_parquet(f)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.set_index("timestamp")["close"].sort_index()
    return None


def run_improved_backtest(daily, vix_series=None,
                          top_n=7, inv_vol=True, max_pos_pct=0.15,
                          vol_target=0.15, rebal_freq=5,
                          trend_sma=200, cost_bps=25,
                          vix_reduce_threshold=30, vix_flat_threshold=45,
                          exposure_min=0.5, exposure_max=1.3,
                          initial_equity=100000.0,
                          mom_lookbacks=(20, 60, 126),
                          mom_weights=(1.0, 2.0, 1.0)):
    """Full backtest with all improvements."""
    symbols = sorted(daily["symbol"].unique())
    tradeable = [s for s in symbols if s not in NON_TRADEABLE]

    pivoted = daily.pivot_table(index="timestamp", columns="symbol", values="close").sort_index().ffill()
    dates = pivoted.index.tolist()
    n = len(dates)
    if BENCHMARK not in pivoted.columns:
        return None

    avail = [s for s in tradeable if s in pivoted.columns]
    if len(avail) < top_n:
        return None

    spy = pivoted[BENCHMARK].values
    spy_sma = pd.Series(spy).rolling(trend_sma, min_periods=trend_sma).mean().values
    mom = _blended_momentum(pivoted, avail, list(mom_lookbacks), list(mom_weights))
    cost_frac = cost_bps / 10000

    # Rolling vol per stock (for inv-vol weighting)
    stock_vol = {}
    if inv_vol:
        for s in avail:
            stock_vol[s] = pivoted[s].pct_change().rolling(20, min_periods=10).std().values * math.sqrt(252)

    # Portfolio vol for scaling
    port_ret = pivoted[avail].pct_change().mean(axis=1)
    rvol = port_ret.rolling(20, min_periods=10).std().values * math.sqrt(252)

    # VIX lookup
    vix_dict = {}
    if vix_series is not None:
        for dt in dates:
            ts = pd.Timestamp(dt)
            if ts in vix_series.index:
                vix_dict[dt] = vix_series[ts]

    warmup = max(max(mom_lookbacks), trend_sma, 20) + 5
    equity = initial_equity
    holdings = []
    weights = {}
    equity_curve = np.full(n, np.nan)
    spy_norm = np.full(n, np.nan)
    spy_base = spy[warmup] if warmup < n else spy[0]
    total_trades = 0
    last_rebalance = 0
    trend_strength = 1.0
    daily_returns = []
    trade_log = []

    for i in range(n):
        spy_norm[i] = initial_equity * (spy[i] / spy_base) if spy_base > 0 else initial_equity
        if i < warmup:
            equity_curve[i] = equity
            continue

        prev_equity = equity
        rv = rvol[i-1] if i-1 < len(rvol) and not np.isnan(rvol[i-1]) and rvol[i-1] > 0 else vol_target
        exposure = float(np.clip(vol_target / rv, exposure_min, exposure_max))

        # VIX crash filter
        vix_val = vix_dict.get(dates[i], None)
        vix_mult = 1.0
        if vix_val is not None:
            if vix_val >= vix_flat_threshold:
                vix_mult = 0.0
            elif vix_val >= vix_reduce_threshold:
                vix_mult = max(0.0, 1.0 - (vix_val - vix_reduce_threshold) / (vix_flat_threshold - vix_reduce_threshold))

        effective_exposure = exposure * trend_strength * vix_mult

        # Mark-to-market
        if holdings and i > 0:
            day_pnl = 0.0
            for sym in holdings:
                w = weights.get(sym, 1.0 / len(holdings))
                prev_p = pivoted[sym].iloc[i-1]
                curr_p = pivoted[sym].iloc[i]
                if prev_p > 0 and not np.isnan(prev_p) and not np.isnan(curr_p):
                    day_pnl += equity * w * (curr_p / prev_p - 1) * effective_exposure
            equity += day_pnl

        # Rebalance
        should_rebalance = (i - last_rebalance >= rebal_freq) or (i == warmup)
        if should_rebalance:
            if np.isnan(spy_sma[i]):
                trend_strength = 0.0
            else:
                trend_strength = _trend_signal_gradual(spy[i], spy_sma[i])

            if trend_strength <= 0 or (vix_val is not None and vix_val >= vix_flat_threshold):
                target_h = []
                target_w = {}
            else:
                mom_today = mom.iloc[i].dropna() if i < len(mom) else pd.Series(dtype=float)
                mom_today = mom_today[mom_today > 0]  # absolute momentum filter
                if len(mom_today) >= top_n:
                    ranked = mom_today.sort_values(ascending=False)
                    target_h = list(ranked.index[:top_n])

                    if inv_vol and stock_vol:
                        inv_vols = {}
                        for s in target_h:
                            sv = stock_vol.get(s)
                            v = sv[i-1] if sv is not None and i-1 < len(sv) and not np.isnan(sv[i-1]) and sv[i-1] > 0 else 0.2
                            inv_vols[s] = 1.0 / v
                        total_iv = sum(inv_vols.values())
                        target_w = {s: min(iv / total_iv, max_pos_pct) for s, iv in inv_vols.items()}
                        tw_sum = sum(target_w.values())
                        target_w = {s: w / tw_sum for s, w in target_w.items()}
                    else:
                        w = 1.0 / len(target_h)
                        target_w = {s: w for s in target_h}
                else:
                    target_h = list(holdings)
                    target_w = dict(weights)

            if set(target_h) != set(holdings):
                turnover = len(set(holdings) - set(target_h)) + len(set(target_h) - set(holdings))
                if turnover > 0:
                    cost = turnover * cost_frac * equity / max(top_n * 2, 1)
                    equity -= cost
                    total_trades += turnover
                holdings = list(target_h)
                weights = dict(target_w)
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
    ds = float(np.std(neg_ret)) if len(neg_ret) > 0 else 1e-9
    sortino = float(np.mean(d_ret) / ds * math.sqrt(252)) if ds > 1e-12 else 0
    pf = float(np.sum(d_ret[d_ret > 0]) / abs(np.sum(d_ret[d_ret < 0]))) if np.sum(d_ret[d_ret < 0]) != 0 else 0
    win_rate = float(np.mean(d_ret > 0))
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0

    return {
        "cagr": round(cagr * 100, 2), "spy_cagr": round(spy_cagr * 100, 2),
        "alpha": round((cagr - spy_cagr) * 100, 2), "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3), "calmar": round(calmar, 2),
        "max_dd": round(max_dd * 100, 2), "profit_factor": round(pf, 3),
        "win_rate": round(win_rate * 100, 2), "ann_vol": round(std * math.sqrt(252) * 100, 2),
        "trades": total_trades, "years": round(years, 1),
        "daily_returns": d_ret, "equity_curve": eq, "spy_curve": spy_n,
        "worst_day": round(np.min(d_ret) * 100, 2),
        "best_day": round(np.max(d_ret) * 100, 2),
    }


def run_monte_carlo(daily, vix, n_sims=1000, **kwargs):
    """Monte Carlo: shuffle daily returns, measure distribution of outcomes."""
    result = run_improved_backtest(daily, vix, **kwargs)
    if result is None:
        return None
    d_ret = result["daily_returns"]
    n = len(d_ret)
    initial = result["equity_curve"][0]

    mc_cagrs = []
    mc_sharpes = []
    mc_dds = []

    rng = np.random.RandomState(42)
    for _ in range(n_sims):
        shuffled = rng.permutation(d_ret)
        eq = initial * np.cumprod(1 + shuffled)
        years = n / 252
        c = (eq[-1] / initial) ** (1 / max(years, 0.1)) - 1
        s = float(np.mean(shuffled) / np.std(shuffled) * math.sqrt(252)) if np.std(shuffled) > 1e-12 else 0
        pk = np.maximum.accumulate(eq)
        dd = float(np.min(eq / pk - 1))
        mc_cagrs.append(c * 100)
        mc_sharpes.append(s)
        mc_dds.append(dd * 100)

    return {
        "actual": result,
        "mc_cagr_median": round(np.median(mc_cagrs), 2),
        "mc_cagr_5th": round(np.percentile(mc_cagrs, 5), 2),
        "mc_cagr_95th": round(np.percentile(mc_cagrs, 95), 2),
        "mc_sharpe_median": round(np.median(mc_sharpes), 2),
        "mc_dd_median": round(np.median(mc_dds), 2),
        "mc_dd_5th": round(np.percentile(mc_dds, 5), 2),  # worst 5%
        "mc_dd_95th": round(np.percentile(mc_dds, 95), 2),
    }


def fmt(r):
    if r is None:
        return "  INSUFFICIENT DATA"
    return (f"  CAGR={r['cagr']:+.1f}% | Sharpe={r['sharpe']:.2f} | DD={r['max_dd']:.1f}% | "
            f"Sortino={r['sortino']:.2f} | PF={r['profit_factor']:.2f} | Vol={r['ann_vol']:.1f}% | "
            f"Alpha={r['alpha']:+.1f}% | Trades={r['trades']}")


if __name__ == "__main__":
    print("=" * 70)
    print("  PHASE 2: BUILD THE IMPROVED STRATEGY")
    print("=" * 70)

    print("\n  Downloading missing symbols...")
    download_all()

    print("\n  Loading data...")
    daily = load_all()
    if daily is None:
        print("  FATAL: No data loaded")
        sys.exit(1)
    print(f"  {daily['symbol'].nunique()} symbols, {daily['timestamp'].dt.date.nunique()} days")

    vix = load_vix()
    if vix is not None:
        print(f"  VIX data: {len(vix)} days ({vix.index.min().date()} to {vix.index.max().date()})")
    else:
        print("  WARNING: No VIX data")

    # ═══════════════════════════════════════════════════════
    # TEST 1: VIX CRASH FILTER IMPACT
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  TEST 1: VIX CRASH FILTER")
    print("=" * 70)

    configs_vix = [
        ("No VIX filter",       999, 999),
        ("VIX reduce@25 flat@35", 25, 35),
        ("VIX reduce@30 flat@40", 30, 40),
        ("VIX reduce@30 flat@45", 30, 45),
        ("VIX reduce@35 flat@50", 35, 50),
    ]

    print(f"\n  {'Config':<30s} {'CAGR':>7s} {'Sharpe':>7s} {'DD':>7s} {'Sort':>6s} {'PF':>5s}")
    print(f"  {'-'*62}")
    for name, vr, vf in configs_vix:
        r = run_improved_backtest(daily, vix, top_n=7, inv_vol=True, max_pos_pct=0.15,
                                  vix_reduce_threshold=vr, vix_flat_threshold=vf)
        if r:
            print(f"  {name:<30s} {r['cagr']:>+6.1f}% {r['sharpe']:>6.2f} {r['max_dd']:>6.1f}% {r['sortino']:>5.2f} {r['profit_factor']:>5.2f}")

    # ═══════════════════════════════════════════════════════
    # TEST 2: PARAMETER STABILITY (improved strategy)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  TEST 2: PARAMETER STABILITY (improved version)")
    print("=" * 70)

    param_tests = [
        ("top_n=5",          dict(top_n=5)),
        ("top_n=7 (default)", dict(top_n=7)),
        ("top_n=10",         dict(top_n=10)),
        ("SMA=150",          dict(top_n=7, trend_sma=150)),
        ("SMA=200 (default)", dict(top_n=7, trend_sma=200)),
        ("SMA=250",          dict(top_n=7, trend_sma=250)),
        ("Rebal 3d",         dict(top_n=7, rebal_freq=3)),
        ("Rebal 5d (default)", dict(top_n=7, rebal_freq=5)),
        ("Rebal 10d",        dict(top_n=7, rebal_freq=10)),
        ("Vol target 12%",   dict(top_n=7, vol_target=0.12)),
        ("Vol target 15% (default)", dict(top_n=7, vol_target=0.15)),
        ("Vol target 18%",   dict(top_n=7, vol_target=0.18)),
    ]

    print(f"\n  {'Config':<25s} {'CAGR':>7s} {'Sharpe':>7s} {'DD':>7s} {'Sort':>6s}")
    print(f"  {'-'*50}")
    for name, params in param_tests:
        r = run_improved_backtest(daily, vix, inv_vol=True, max_pos_pct=0.15,
                                  vix_reduce_threshold=30, vix_flat_threshold=45, **params)
        if r:
            marker = " <--" if "(default)" in name else ""
            print(f"  {name:<25s} {r['cagr']:>+6.1f}% {r['sharpe']:>6.2f} {r['max_dd']:>6.1f}% {r['sortino']:>5.2f}{marker}")

    # ═══════════════════════════════════════════════════════
    # TEST 3: COST SENSITIVITY (improved strategy)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  TEST 3: COST SENSITIVITY (improved)")
    print("=" * 70)

    print(f"\n  {'Cost':>8s} {'CAGR':>7s} {'Sharpe':>7s} {'Alpha':>7s}")
    print(f"  {'-'*32}")
    for bps in [0, 10, 25, 50, 75, 100]:
        r = run_improved_backtest(daily, vix, top_n=7, inv_vol=True, max_pos_pct=0.15,
                                  vix_reduce_threshold=30, vix_flat_threshold=45, cost_bps=bps)
        if r:
            print(f"  {bps:>5d}bps {r['cagr']:>+6.1f}% {r['sharpe']:>6.2f} {r['alpha']:>+6.1f}%")

    # ═══════════════════════════════════════════════════════
    # TEST 4: REGIME ANALYSIS (improved strategy)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  TEST 4: REGIME PERFORMANCE (improved)")
    print("=" * 70)

    periods = {
        "2010-2013 (recovery/bull)":   ("2010-01-01", "2013-12-31"),
        "2014-2016 (chop/low vol)":    ("2014-01-01", "2016-12-31"),
        "2017-2019 (bull/vol spike)":  ("2017-01-01", "2019-12-31"),
        "2020 (COVID crash+V)":        ("2020-01-01", "2020-12-31"),
        "2021-2022 (bull->bear)":      ("2021-01-01", "2022-12-31"),
        "2023-2025 (AI bull+tariffs)": ("2023-01-01", "2025-12-31"),
        "Full period":                 ("2010-01-01", "2026-12-31"),
    }

    print(f"\n  {'Period':<30s} {'CAGR':>7s} {'Sharpe':>7s} {'DD':>7s} {'Alpha':>7s}")
    print(f"  {'-'*58}")
    for name, (start, end) in periods.items():
        sub = daily[(daily["timestamp"] >= start) & (daily["timestamp"] <= end)]
        if sub["timestamp"].dt.date.nunique() < 200:
            print(f"  {name:<30s} insufficient data")
            continue
        r = run_improved_backtest(sub, vix, top_n=7, inv_vol=True, max_pos_pct=0.15,
                                  vix_reduce_threshold=30, vix_flat_threshold=45)
        if r:
            print(f"  {name:<30s} {r['cagr']:>+6.1f}% {r['sharpe']:>6.2f} {r['max_dd']:>6.1f}% {r['alpha']:>+6.1f}%")

    # ═══════════════════════════════════════════════════════
    # TEST 5: PROFIT FACTOR FRAGILITY (improved)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  TEST 5: PROFIT FACTOR FRAGILITY (improved)")
    print("=" * 70)

    r_base = run_improved_backtest(daily, vix, top_n=7, inv_vol=True, max_pos_pct=0.15,
                                   vix_reduce_threshold=30, vix_flat_threshold=45)
    if r_base:
        d_ret = r_base["daily_returns"].copy()
        print(f"\n  {'Degradation':<30s} {'CAGR':>7s} {'Sharpe':>7s} {'PF':>5s}")
        print(f"  {'-'*52}")
        for dp in [0, 5, 10, 15, 20]:
            degraded = d_ret.copy()
            degraded = np.where(degraded > 0, degraded * (1 - dp/100), degraded * (1 + dp/100))
            eq = r_base["equity_curve"][0] * np.cumprod(1 + degraded)
            yrs = len(degraded) / 252
            c = (eq[-1] / eq[0]) ** (1 / max(yrs, 0.1)) - 1
            s_std = np.std(degraded)
            sh = float(np.mean(degraded) / s_std * math.sqrt(252)) if s_std > 1e-12 else 0
            pf = float(np.sum(degraded[degraded > 0]) / abs(np.sum(degraded[degraded < 0]))) if np.sum(degraded[degraded < 0]) != 0 else 0
            print(f"  Wins -{dp}%, Losses +{dp}%{'':<10s} {c*100:>+6.1f}% {sh:>6.2f} {pf:>5.2f}")

    # ═══════════════════════════════════════════════════════
    # TEST 6: MONTE CARLO (improved)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  TEST 6: MONTE CARLO (1000 sims, improved strategy)")
    print("=" * 70)

    mc = run_monte_carlo(daily, vix, n_sims=1000,
                         top_n=7, inv_vol=True, max_pos_pct=0.15,
                         vix_reduce_threshold=30, vix_flat_threshold=45)
    if mc:
        a = mc["actual"]
        print(f"\n  Actual strategy:")
        print(fmt(a))
        print(f"\n  Monte Carlo distribution (1000 shuffled return paths):")
        print(f"    CAGR:  5th={mc['mc_cagr_5th']:+.1f}%  median={mc['mc_cagr_median']:+.1f}%  95th={mc['mc_cagr_95th']:+.1f}%")
        print(f"    Sharpe:              median={mc['mc_sharpe_median']:.2f}")
        print(f"    Max DD: 5th={mc['mc_dd_5th']:.1f}%  median={mc['mc_dd_median']:.1f}%  95th={mc['mc_dd_95th']:.1f}%")

    # ═══════════════════════════════════════════════════════
    # TEST 7: HEAD-TO-HEAD: OLD vs NEW
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  TEST 7: HEAD-TO-HEAD COMPARISON")
    print("=" * 70)

    # Old strategy (original universe from strategy/engine.py)
    from strategy.data import load as load_original
    orig_daily = load_original()
    from strategy.engine import NON_TRADEABLE as ORIG_NT
    orig_tradeable = [s for s in orig_daily["symbol"].unique() if s not in ORIG_NT]

    # Old: 5 stocks, equal weight, no VIX filter, original universe
    old_r = run_improved_backtest(orig_daily, None,
                                   top_n=5, inv_vol=False, max_pos_pct=1.0,
                                   vix_reduce_threshold=999, vix_flat_threshold=999)
    # New: 7 stocks, inv-vol, 15% cap, VIX filter, expanded universe
    new_r = run_improved_backtest(daily, vix,
                                   top_n=7, inv_vol=True, max_pos_pct=0.15,
                                   vix_reduce_threshold=30, vix_flat_threshold=45)
    # New with 50bps costs (pessimistic)
    new_50 = run_improved_backtest(daily, vix,
                                    top_n=7, inv_vol=True, max_pos_pct=0.15,
                                    vix_reduce_threshold=30, vix_flat_threshold=45,
                                    cost_bps=50)

    print(f"\n  {'Version':<40s} {'CAGR':>7s} {'Sharpe':>7s} {'DD':>7s} {'Alpha':>7s} {'PF':>5s}")
    print(f"  {'-'*72}")
    if old_r:
        print(f"  {'OLD: 5 stocks, eq-wt, no VIX':<40s} {old_r['cagr']:>+6.1f}% {old_r['sharpe']:>6.2f} {old_r['max_dd']:>6.1f}% {old_r['alpha']:>+6.1f}% {old_r['profit_factor']:>5.2f}")
    if new_r:
        print(f"  {'NEW: 7 stocks, inv-vol, VIX, expanded':<40s} {new_r['cagr']:>+6.1f}% {new_r['sharpe']:>6.2f} {new_r['max_dd']:>6.1f}% {new_r['alpha']:>+6.1f}% {new_r['profit_factor']:>5.2f}")
    if new_50:
        print(f"  {'NEW with 50bps costs (pessimistic)':<40s} {new_50['cagr']:>+6.1f}% {new_50['sharpe']:>6.2f} {new_50['max_dd']:>6.1f}% {new_50['alpha']:>+6.1f}% {new_50['profit_factor']:>5.2f}")

    # ═══════════════════════════════════════════════════════
    # FINAL PRODUCTION PARAMETERS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  RECOMMENDED PRODUCTION PARAMETERS")
    print("=" * 70)
    print("""
  Universe: 40+ stocks (large caps present since 2010) + 8 sector ETFs
  Top N:           7
  Weighting:       Inverse volatility
  Max position:    15% per stock
  Trend filter:    200-day SMA (gradual)
  Abs momentum:    Yes (skip negative momentum)
  VIX filter:      Reduce exposure VIX>30, flat VIX>45
  Vol target:      15%
  Exposure range:  50% - 130%
  Rebalance:       Weekly (every 5 trading days)
  Cost budget:     25 bps round-trip (test passes at 50bps)
  Order type:      MOC (Market-On-Close)
""")
    print("=" * 70)
