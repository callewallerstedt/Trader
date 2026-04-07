#!/usr/bin/env python3
"""
Targeted research: increase profit factor without increasing drawdown.

Approaches tested:
1. Momentum quality (trend R²) - select stocks with SMOOTH uptrends
2. Volatility-adjusted momentum (momentum / volatility)
3. Momentum breadth (how many timeframes agree)
4. Longer rebalance periods (15d, 20d)
5. Holding period constraint (minimum hold before selling)
6. Gradual VIX (already implemented, verify vs binary)
7. Execution stress (50-75bps, random slippage)
8. Factor decomposition (which component adds Sharpe)
9. VIX filter in crash windows (does it remove rebounds?)
"""
import sys, math, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy import stats
from strategy.engine import (
    UNIVERSE, NON_TRADEABLE, VIX_SYMBOL,
    _blended_momentum, _trend_signal_gradual, _vix_exposure_multiplier,
    Config,
)
from strategy.data import load, load_vix


def _trend_r_squared(pivoted, symbols, lookback=60):
    """
    Compute R² of log-price trend for each stock.
    High R² = smooth/consistent trend (quality signal).
    """
    r2_df = pd.DataFrame(index=pivoted.index, columns=symbols, dtype=float)
    for sym in symbols:
        if sym not in pivoted.columns:
            continue
        log_p = np.log(pivoted[sym].replace(0, np.nan))
        for i in range(lookback, len(pivoted)):
            window = log_p.iloc[i-lookback:i].dropna()
            if len(window) < lookback * 0.8:
                continue
            x = np.arange(len(window))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, window.values)
            r2_df.iloc[i][sym] = r_value ** 2
    return r2_df


def _vol_adjusted_momentum(pivoted, symbols, lookbacks, weights, vol_lb=20):
    """Momentum divided by rolling volatility = signal-to-noise ratio."""
    mom = _blended_momentum(pivoted, symbols, lookbacks, weights)
    for sym in symbols:
        if sym not in pivoted.columns:
            continue
        vol = pivoted[sym].pct_change().rolling(vol_lb, min_periods=10).std()
        vol = vol.replace(0, np.nan)
        mom[sym] = mom[sym] / vol
    return mom


def _momentum_breadth(pivoted, symbols):
    """Count how many timeframes show positive momentum (1m, 3m, 6m)."""
    m1 = pivoted[symbols].pct_change(21)
    m3 = pivoted[symbols].pct_change(63)
    m6 = pivoted[symbols].pct_change(126)
    breadth = (m1 > 0).astype(float) + (m3 > 0).astype(float) + (m6 > 0).astype(float)
    return breadth / 3.0  # 0 to 1


def run_backtest_v3(daily, vix_series=None,
                    top_n=10, inv_vol=True, max_pos_pct=0.15,
                    vol_target=0.15, rebal_freq=10,
                    trend_sma=200, cost_bps=25,
                    vix_reduce=25, vix_flat=35,
                    exposure_min=0.5, exposure_max=1.3,
                    initial_equity=100000.0,
                    mom_lookbacks=(20, 60, 126),
                    mom_weights=(1.0, 2.0, 1.0),
                    use_quality=False, quality_lookback=60,
                    use_vol_adj_mom=False,
                    use_breadth=False,
                    min_hold_days=0,
                    random_slippage_bps=0,
                    label=""):
    """Backtest with optional quality factor and other enhancements."""
    symbols = sorted(daily["symbol"].unique())
    tradeable = [s for s in symbols if s not in NON_TRADEABLE]
    pivoted = daily.pivot_table(index="timestamp", columns="symbol", values="close").sort_index().ffill()
    dates = pivoted.index.tolist()
    n = len(dates)
    if "SPY" not in pivoted.columns:
        return None

    avail = [s for s in tradeable if s in pivoted.columns]
    if len(avail) < top_n:
        return None

    spy = pivoted["SPY"].values
    spy_sma = pd.Series(spy).rolling(trend_sma, min_periods=trend_sma).mean().values

    # Momentum signal
    if use_vol_adj_mom:
        mom = _vol_adjusted_momentum(pivoted, avail, list(mom_lookbacks), list(mom_weights))
    else:
        mom = _blended_momentum(pivoted, avail, list(mom_lookbacks), list(mom_weights))

    # Quality signal (trend R²)
    quality = None
    if use_quality:
        quality = _trend_r_squared(pivoted, avail, lookback=quality_lookback)

    # Breadth signal
    breadth = None
    if use_breadth:
        breadth = _momentum_breadth(pivoted, avail)

    # Stock vol for inv-vol weighting
    stock_vol = {}
    if inv_vol:
        for s in avail:
            stock_vol[s] = pivoted[s].pct_change().rolling(20, min_periods=10).std().values * math.sqrt(252)

    # Portfolio vol
    port_ret = pivoted[avail].pct_change().mean(axis=1)
    rvol = port_ret.rolling(20, min_periods=10).std().values * math.sqrt(252)

    # VIX
    vix_dict = {}
    if vix_series is not None:
        for dt in dates:
            ts = pd.Timestamp(dt)
            if ts in vix_series.index:
                vix_dict[dt] = float(vix_series[ts])

    warmup = max(max(mom_lookbacks), trend_sma, 20, quality_lookback if use_quality else 0) + 10
    equity = initial_equity
    holdings = []
    weights = {}
    entry_day = {}  # sym -> day entered (for min hold)
    equity_curve = np.full(n, np.nan)
    spy_norm = np.full(n, np.nan)
    spy_base = spy[warmup] if warmup < n else spy[0]
    total_trades = 0
    last_rebalance = 0
    trend_strength = 1.0
    cost_frac = cost_bps / 10000
    rng = np.random.RandomState(42) if random_slippage_bps > 0 else None

    for i in range(n):
        spy_norm[i] = initial_equity * (spy[i] / spy_base) if spy_base > 0 else initial_equity
        if i < warmup:
            equity_curve[i] = equity
            continue

        rv = rvol[i-1] if i-1 < len(rvol) and not np.isnan(rvol[i-1]) and rvol[i-1] > 0 else vol_target
        exposure = float(np.clip(vol_target / rv, exposure_min, exposure_max))
        vix_val = vix_dict.get(dates[i])
        cfg = Config(vix_reduce_threshold=vix_reduce, vix_flat_threshold=vix_flat)
        vix_mult = _vix_exposure_multiplier(vix_val, cfg)
        effective_exposure = exposure * trend_strength * vix_mult

        if holdings and i > 0:
            day_pnl = 0.0
            for sym in holdings:
                w = weights.get(sym, 1.0 / len(holdings))
                prev_p = pivoted[sym].iloc[i-1]
                curr_p = pivoted[sym].iloc[i]
                if prev_p > 0 and not np.isnan(prev_p) and not np.isnan(curr_p):
                    day_pnl += equity * w * (curr_p / prev_p - 1) * effective_exposure
            equity += day_pnl

        should_rebalance = (i - last_rebalance >= rebal_freq) or (i == warmup)
        if should_rebalance:
            if np.isnan(spy_sma[i]):
                trend_strength = 0.0
            else:
                trend_strength = _trend_signal_gradual(spy[i], spy_sma[i])

            if trend_strength <= 0 or vix_mult <= 0:
                target_h = []
                target_w = {}
            else:
                mom_today = mom.iloc[i].dropna() if i < len(mom) else pd.Series(dtype=float)
                mom_today = mom_today[mom_today > 0]  # absolute momentum filter

                if use_quality and quality is not None:
                    q_today = quality.iloc[i].dropna() if i < len(quality) else pd.Series(dtype=float)
                    # Composite score = momentum * quality
                    common = mom_today.index.intersection(q_today.index)
                    if len(common) > 0:
                        composite = mom_today[common] * q_today[common]
                        composite = composite.dropna()
                        if len(composite) >= top_n:
                            mom_today = composite

                if use_breadth and breadth is not None:
                    b_today = breadth.iloc[i].dropna() if i < len(breadth) else pd.Series(dtype=float)
                    common = mom_today.index.intersection(b_today.index)
                    if len(common) > 0:
                        # Boost momentum by breadth (0.5 to 1.5x multiplier)
                        mom_today = mom_today[common] * (0.5 + b_today[common])

                if len(mom_today) >= top_n:
                    ranked = mom_today.sort_values(ascending=False)

                    # Min hold constraint: keep current holdings if < min_hold_days
                    if min_hold_days > 0 and holdings:
                        candidates = list(ranked.index[:top_n * 2])
                        forced_holds = [s for s in holdings if (i - entry_day.get(s, 0)) < min_hold_days and s in ranked.index]
                        target_h = list(forced_holds)
                        for s in candidates:
                            if s not in target_h and len(target_h) < top_n:
                                target_h.append(s)
                        target_h = target_h[:top_n]
                    else:
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
                        target_w = {s: w / tw_sum for s, w in target_w.items()} if tw_sum > 0 else {s: 1.0/len(target_h) for s in target_h}
                    else:
                        w = 1.0 / len(target_h)
                        target_w = {s: w for s in target_h}
                else:
                    target_h = list(holdings)
                    target_w = dict(weights)

            if set(target_h) != set(holdings):
                turnover = len(set(holdings) - set(target_h)) + len(set(target_h) - set(holdings))
                if turnover > 0:
                    # Apply random slippage
                    slippage = 0
                    if rng is not None and random_slippage_bps > 0:
                        slippage = rng.uniform(0, random_slippage_bps / 10000) * equity * turnover / max(top_n * 2, 1)
                    cost = turnover * cost_frac * equity / max(top_n * 2, 1) + slippage
                    equity -= cost
                    total_trades += turnover
                    for s in set(target_h) - set(holdings):
                        entry_day[s] = i
                holdings = list(target_h)
                weights = dict(target_w)
                last_rebalance = i

        equity_curve[i] = equity

    eq = equity_curve[warmup:]
    spy_n = spy_norm[warmup:]
    valid = ~np.isnan(eq) & ~np.isnan(spy_n)
    eq, spy_n = eq[valid], spy_n[valid]

    if len(eq) < 252:
        return None

    d_ret = np.diff(eq) / eq[:-1]
    d_ret = d_ret[np.isfinite(d_ret)]
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

    return {
        "label": label, "cagr": round(cagr * 100, 2),
        "spy_cagr": round(spy_cagr * 100, 2),
        "alpha": round((cagr - spy_cagr) * 100, 2),
        "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
        "max_dd": round(max_dd * 100, 2),
        "pf": round(pf, 3), "win_rate": round(win_rate * 100, 2),
        "trades": total_trades, "years": round(years, 1),
        "ann_vol": round(std * math.sqrt(252) * 100, 2),
        "daily_returns": d_ret, "equity_curve": eq, "spy_curve": spy_n,
    }


def pf_fragility(result, degrade_pcts=(0, 5, 10)):
    """Test how PF degrades under execution stress."""
    if result is None:
        return []
    d_ret = result["daily_returns"]
    out = []
    for dp in degrade_pcts:
        degraded = np.where(d_ret > 0, d_ret * (1 - dp/100), d_ret * (1 + dp/100))
        std = np.std(degraded)
        sh = float(np.mean(degraded) / std * math.sqrt(252)) if std > 1e-12 else 0
        pf = float(np.sum(degraded[degraded > 0]) / abs(np.sum(degraded[degraded < 0]))) if np.sum(degraded[degraded < 0]) != 0 else 0
        out.append((dp, sh, pf))
    return out


def row(r, extra_cols=""):
    if r is None:
        return f"  {'(insufficient data)':<45s}"
    return (f"  {r['label']:<45s} {r['cagr']:>+6.1f}% {r['sharpe']:>6.2f} {r['max_dd']:>6.1f}% "
            f"{r['sortino']:>5.2f} {r['pf']:>5.3f} {r['trades']:>5d} {r['ann_vol']:>5.1f}%{extra_cols}")


if __name__ == "__main__":
    print("Loading data...")
    daily = load()
    vix = load_vix()
    print(f"  {daily['symbol'].nunique()} symbols, {daily['timestamp'].dt.date.nunique()} days\n")

    HEADER = f"  {'Config':<45s} {'CAGR':>7s} {'Sharpe':>7s} {'DD':>7s} {'Sort':>5s} {'PF':>7s} {'Trd':>5s} {'Vol':>5s}"
    DIVIDER = f"  {'-'*90}"

    # ═══════════════════════════════════════════════════════
    # BASELINE
    # ═══════════════════════════════════════════════════════
    print("=" * 90)
    print("  BASELINE (current production V2)")
    print("=" * 90)
    print(HEADER)
    print(DIVIDER)
    baseline = run_backtest_v3(daily, vix, label="V2 baseline (N=10, 10d, VIX 25/35)")
    print(row(baseline))

    # ═══════════════════════════════════════════════════════
    # 1. QUALITY FACTOR (Trend R²)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("  1. QUALITY FACTOR: Momentum * Trend R² (smooth uptrends)")
    print("=" * 90)
    print(HEADER)
    print(DIVIDER)

    for qlb in [40, 60, 90, 126]:
        r = run_backtest_v3(daily, vix, use_quality=True, quality_lookback=qlb,
                            label=f"Mom * R²(lb={qlb})")
        print(row(r))

    # ═══════════════════════════════════════════════════════
    # 2. VOLATILITY-ADJUSTED MOMENTUM
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("  2. VOL-ADJUSTED MOMENTUM (momentum / volatility)")
    print("=" * 90)
    print(HEADER)
    print(DIVIDER)
    r = run_backtest_v3(daily, vix, use_vol_adj_mom=True, label="Vol-adj momentum")
    print(row(r))

    # ═══════════════════════════════════════════════════════
    # 3. MOMENTUM BREADTH
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("  3. MOMENTUM BREADTH (boost if multiple timeframes agree)")
    print("=" * 90)
    print(HEADER)
    print(DIVIDER)
    r = run_backtest_v3(daily, vix, use_breadth=True, label="Mom * breadth")
    print(row(r))

    # ═══════════════════════════════════════════════════════
    # 4. COMBINED: Quality + Breadth
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("  4. COMBINED: Quality + Breadth")
    print("=" * 90)
    print(HEADER)
    print(DIVIDER)
    for qlb in [60, 90]:
        r = run_backtest_v3(daily, vix, use_quality=True, quality_lookback=qlb,
                            use_breadth=True, label=f"Quality(R²={qlb}) + Breadth")
        print(row(r))

    # ═══════════════════════════════════════════════════════
    # 5. LONGER REBALANCE
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("  5. LONGER REBALANCE PERIODS")
    print("=" * 90)
    print(HEADER)
    print(DIVIDER)
    for freq in [10, 15, 20, 30]:
        r = run_backtest_v3(daily, vix, rebal_freq=freq, label=f"Rebal every {freq}d")
        print(row(r))

    # ═══════════════════════════════════════════════════════
    # 6. MIN HOLDING PERIOD
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("  6. MINIMUM HOLDING PERIOD")
    print("=" * 90)
    print(HEADER)
    print(DIVIDER)
    for mh in [0, 10, 20, 30]:
        r = run_backtest_v3(daily, vix, min_hold_days=mh, label=f"Min hold {mh}d")
        print(row(r))

    # ═══════════════════════════════════════════════════════
    # 7. VIX FILTER IN CRASH WINDOWS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("  7. VIX FILTER: CRASH WINDOW ANALYSIS")
    print("=" * 90)
    crash_windows = [
        ("COVID (2020)",     "2020-01-01", "2020-12-31"),
        ("2022 bear",        "2022-01-01", "2022-12-31"),
        ("2018 vol spike",   "2018-01-01", "2018-12-31"),
        ("2025 tariff shock","2025-01-01", "2025-12-31"),
    ]
    print(f"\n  {'Window':<22s} {'With VIX':>24s} {'Without VIX':>24s} {'VIX better?':>14s}")
    print(f"  {'-'*85}")
    for name, s, e in crash_windows:
        sub = daily[(daily["timestamp"] >= s) & (daily["timestamp"] <= e)]
        if sub["timestamp"].dt.date.nunique() < 100:
            print(f"  {name:<22s} insufficient data")
            continue
        with_vix = run_backtest_v3(sub, vix, label=f"VIX on: {name}")
        without_vix = run_backtest_v3(sub, vix, vix_reduce=999, vix_flat=999, label=f"VIX off: {name}")
        if with_vix and without_vix:
            v_str = f"CAGR={with_vix['cagr']:+.1f}% DD={with_vix['max_dd']:.1f}%"
            nv_str = f"CAGR={without_vix['cagr']:+.1f}% DD={without_vix['max_dd']:.1f}%"
            better = "YES" if with_vix['max_dd'] > without_vix['max_dd'] else "NO"
            print(f"  {name:<22s} {v_str:>24s} {nv_str:>24s} {better:>14s}")

    # ═══════════════════════════════════════════════════════
    # 8. EXECUTION STRESS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("  8. EXECUTION STRESS TEST")
    print("=" * 90)
    print(HEADER)
    print(DIVIDER)
    for bps, slip in [(25, 0), (50, 0), (75, 0), (25, 10), (25, 25), (50, 25)]:
        r = run_backtest_v3(daily, vix, cost_bps=bps, random_slippage_bps=slip,
                            label=f"Cost={bps}bps + slippage={slip}bps")
        print(row(r))

    # ═══════════════════════════════════════════════════════
    # 9. FACTOR DECOMPOSITION
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("  9. FACTOR DECOMPOSITION (which component adds Sharpe?)")
    print("=" * 90)
    print(HEADER)
    print(DIVIDER)

    # Full strategy
    full = run_backtest_v3(daily, vix, label="Full (mom + trend + VIX + vol)")
    print(row(full))

    # No VIX filter
    no_vix = run_backtest_v3(daily, vix, vix_reduce=999, vix_flat=999,
                              label="No VIX filter")
    print(row(no_vix))

    # No trend filter
    no_trend = run_backtest_v3(daily, vix, trend_sma=1, label="No trend filter (SMA=1)")
    print(row(no_trend))

    # No vol scaling
    no_vol = run_backtest_v3(daily, vix, vol_target=99, exposure_min=1.0, exposure_max=1.0,
                              label="No vol scaling (fixed 100%)")
    print(row(no_vol))

    # Only momentum (no trend, no VIX, no vol scaling)
    pure_mom = run_backtest_v3(daily, vix, trend_sma=1, vix_reduce=999, vix_flat=999,
                                vol_target=99, exposure_min=1.0, exposure_max=1.0,
                                label="Pure momentum only")
    print(row(pure_mom))

    print(f"\n  Contribution analysis:")
    if full and no_vix and no_trend and no_vol and pure_mom:
        print(f"    VIX filter adds:      Sharpe {full['sharpe'] - no_vix['sharpe']:+.3f}, PF {full['pf'] - no_vix['pf']:+.3f}, DD {full['max_dd'] - no_vix['max_dd']:+.1f}%")
        print(f"    Trend filter adds:    Sharpe {full['sharpe'] - no_trend['sharpe']:+.3f}, PF {full['pf'] - no_trend['pf']:+.3f}, DD {full['max_dd'] - no_trend['max_dd']:+.1f}%")
        print(f"    Vol scaling adds:     Sharpe {full['sharpe'] - no_vol['sharpe']:+.3f}, PF {full['pf'] - no_vol['pf']:+.3f}, DD {full['max_dd'] - no_vol['max_dd']:+.1f}%")

    # ═══════════════════════════════════════════════════════
    # 10. BEST COMBINED
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("  10. BEST COMBINATIONS")
    print("=" * 90)
    print(HEADER)
    print(DIVIDER)

    combos = [
        ("Baseline V2",
         dict()),
        ("Quality(R²=60)",
         dict(use_quality=True, quality_lookback=60)),
        ("Quality(R²=90)",
         dict(use_quality=True, quality_lookback=90)),
        ("Quality(R²=60) + 15d rebal",
         dict(use_quality=True, quality_lookback=60, rebal_freq=15)),
        ("Quality(R²=60) + 20d rebal",
         dict(use_quality=True, quality_lookback=60, rebal_freq=20)),
        ("Quality(R²=60) + min hold 10d",
         dict(use_quality=True, quality_lookback=60, min_hold_days=10)),
        ("Quality(R²=60) + breadth + 15d",
         dict(use_quality=True, quality_lookback=60, use_breadth=True, rebal_freq=15)),
        ("Vol-adj mom + 15d rebal",
         dict(use_vol_adj_mom=True, rebal_freq=15)),
        ("Quality(R²=60) + N=7",
         dict(use_quality=True, quality_lookback=60, top_n=7)),
    ]

    best_pf = 0
    best_name = ""
    for name, params in combos:
        r = run_backtest_v3(daily, vix, label=name, **params)
        if r:
            frag = pf_fragility(r, [0, 5, 10])
            frag_str = f"  [5%deg: Sh={frag[1][1]:.2f} PF={frag[1][2]:.2f}]"
            print(row(r, frag_str))
            if r['pf'] > best_pf:
                best_pf = r['pf']
                best_name = name

    # ═══════════════════════════════════════════════════════
    # ANSWER TO KEY QUESTION
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("  ANSWER: How to increase PF without increasing DD")
    print("=" * 90)
    print(f"\n  Best PF achieved: {best_pf:.3f} via '{best_name}'")
    print("=" * 90)
