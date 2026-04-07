#!/usr/bin/env python3
"""
Round 2: Fine-tune the best combination found in alpha_signals.py.

Key findings from round 1:
  - 0.6 mom + 0.4 residual is the best combo
  - N=12, 15d rebalance gives best Sharpe (1.36) and PF (1.275)
  - No vol scaling confirmed
  - PF at 0 cost was 1.295, at 25bps 1.268

This round:
  1. Try different residual momentum lookbacks (60, 90, 126, 180d)
  2. Fine-tune signal weights with N=12
  3. Try non-equal momentum blends for the raw momentum component
  4. VIX filter tuning (per reviewer's request)
  5. Find the path to PF >= 1.30
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
from research.alpha_signals import (
    compute_residual_momentum, compute_momentum_acceleration,
    compute_earnings_drift, compute_combined_score,
    run_backtest, fragility, HEADER, DIV, row,
)


if __name__ == "__main__":
    print("Loading data...")
    daily = load()
    vix = load_vix()
    symbols = sorted(daily["symbol"].unique())
    tradeable = [s for s in symbols if s not in NON_TRADEABLE]
    pivoted = daily.pivot_table(index="timestamp", columns="symbol", values="close").sort_index().ffill()
    print(f"  {len(tradeable)} tradeable, {daily['timestamp'].dt.date.nunique()} days\n")

    # Pre-compute momentum
    print("Computing signals...")
    print("  Blended momentum...", flush=True)
    mom = _blended_momentum(pivoted, tradeable, [20, 60, 126], [1.0, 2.0, 1.0])

    # ═══════════════════════════════════════════════════════
    # 1. RESIDUAL MOMENTUM LOOKBACK SWEEP
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 95}")
    print("  1. RESIDUAL MOMENTUM LOOKBACK SWEEP (N=12, 15d)")
    print("=" * 95)
    print(HEADER)
    print(DIV)

    best_resid = None
    best_resid_pf = 0
    best_resid_lb = 0
    for lb in [40, 60, 90, 126, 180]:
        print(f"  Computing residual {lb}d...", end=" ", flush=True)
        resid = compute_residual_momentum(pivoted, tradeable, lookback=lb)
        if resid is None:
            print("FAILED")
            continue

        # Cross-sectional rank
        mom_rank = mom.rank(axis=1, pct=True)
        resid_rank = resid.rank(axis=1, pct=True)
        sig = mom_rank * 0.6 + resid_rank * 0.4

        r = run_backtest(daily, signal_df=sig, vix_series=vix,
                         top_n=12, rebal_freq=15, use_vol_scaling=False,
                         label=f"Residual {lb}d")
        if r:
            print(row(r))
            if r["pf"] > best_resid_pf:
                best_resid_pf = r["pf"]
                best_resid = resid
                best_resid_lb = lb

    print(f"\n  >>> Best residual lookback: {best_resid_lb}d (PF={best_resid_pf:.3f})")

    # ═══════════════════════════════════════════════════════
    # 2. SIGNAL WEIGHT SWEEP (N=12, 15d, best residual)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 95}")
    print(f"  2. SIGNAL WEIGHT SWEEP (N=12, 15d, residual={best_resid_lb}d)")
    print("=" * 95)
    print(HEADER)
    print(DIV)

    accel = compute_momentum_acceleration(pivoted, tradeable, short_lb=21, long_lb=126)
    mom_rank = mom.rank(axis=1, pct=True)
    resid_rank = best_resid.rank(axis=1, pct=True)
    accel_rank = accel.rank(axis=1, pct=True)

    results = []
    for m_w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for r_w in [0.2, 0.3, 0.4, 0.5]:
            a_w = round(1.0 - m_w - r_w, 1)
            if a_w < 0 or a_w > 0.4:
                continue
            sig = mom_rank * m_w + resid_rank * r_w + accel_rank * a_w
            label = f"M={m_w:.1f} R={r_w:.1f} A={a_w:.1f}"
            r = run_backtest(daily, signal_df=sig, vix_series=vix,
                             top_n=12, rebal_freq=15, use_vol_scaling=False,
                             label=label)
            if r:
                results.append(r)

    results.sort(key=lambda x: x["pf"], reverse=True)
    for r in results[:15]:
        print(row(r))

    print(f"\n  Top 5 by Sharpe:")
    results_sh = sorted(results, key=lambda x: x["sharpe"], reverse=True)
    for r in results_sh[:5]:
        print(row(r))

    best_w = results[0]
    best_w_label = best_w["label"]

    # ═══════════════════════════════════════════════════════
    # 3. N-STOCKS / REBALANCE GRID ON BEST WEIGHTS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 95}")
    print(f"  3. N-STOCKS / REBALANCE GRID (best weights: {best_w_label})")
    print("=" * 95)
    print(HEADER)
    print(DIV)

    best_w_parts = best_w_label.split()
    m_val = float(best_w_parts[0].split("=")[1])
    r_val = float(best_w_parts[1].split("=")[1])
    a_val = float(best_w_parts[2].split("=")[1])
    best_sig = mom_rank * m_val + resid_rank * r_val + accel_rank * a_val

    grid_results = []
    for n in [7, 10, 12, 15]:
        for freq in [10, 15, 20, 25]:
            r = run_backtest(daily, signal_df=best_sig, vix_series=vix,
                             top_n=n, rebal_freq=freq, use_vol_scaling=False,
                             label=f"N={n} rebal={freq}d")
            if r:
                grid_results.append(r)

    grid_results.sort(key=lambda x: x["pf"], reverse=True)
    for r in grid_results[:10]:
        print(row(r))

    top_grid = grid_results[0]

    # ═══════════════════════════════════════════════════════
    # 4. VIX FILTER ANALYSIS (per reviewer's request)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 95}")
    print(f"  4. VIX FILTER ANALYSIS")
    print("=" * 95)
    print(HEADER)
    print(DIV)

    for vr, vf in [(20, 30), (25, 35), (30, 40), (35, 45), (999, 999)]:
        r = run_backtest(daily, signal_df=best_sig, vix_series=vix,
                         top_n=12, rebal_freq=15, use_vol_scaling=False,
                         vix_reduce=vr, vix_flat=vf,
                         label=f"VIX reduce={vr} flat={vf}" if vr < 999 else "No VIX filter")
        if r:
            print(row(r))

    # ═══════════════════════════════════════════════════════
    # 5. MARCH 2020 ANALYSIS (does VIX filter help?)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 95}")
    print(f"  5. MARCH 2020 & 2022 REVERSAL ANALYSIS")
    print("=" * 95)

    crisis_periods = [
        ("COVID crash (2020-01 to 2020-06)", "2020-01-01", "2020-06-30"),
        ("COVID recovery (2020-03 to 2020-12)", "2020-03-01", "2020-12-31"),
        ("2022 bear (2022-01 to 2022-12)", "2022-01-01", "2022-12-31"),
        ("2022 reversal (2022-06 to 2023-06)", "2022-06-01", "2023-06-30"),
    ]
    for name, s, e in crisis_periods:
        sub = daily[(daily["timestamp"] >= s) & (daily["timestamp"] <= e)]
        sub_piv = sub.pivot_table(index="timestamp", columns="symbol", values="close").sort_index().ffill()
        sub_trade = [s for s in tradeable if s in sub_piv.columns]
        if "SPY" in sub_piv.columns and len(sub_piv) > 10:
            spy_vals = sub_piv["SPY"].values
            spy_ret = (spy_vals[-1] / spy_vals[0] - 1) * 100
            print(f"  {name}: SPY {spy_ret:+.1f}%")

            # Compare with/without VIX filter using full data
            for vr, vf, vlabel in [(25, 35, "VIX=25/35"), (999, 999, "No VIX")]:
                r = run_backtest(sub, signal_df=best_sig, vix_series=vix,
                                 top_n=12, rebal_freq=15, use_vol_scaling=False,
                                 vix_reduce=vr, vix_flat=vf,
                                 label=f"  {vlabel}")
                if r:
                    print(f"    {vlabel}: CAGR {r['cagr']:+.1f}% DD {r['max_dd']:.1f}%")
                else:
                    print(f"    {vlabel}: insufficient data for period")
        print()

    # ═══════════════════════════════════════════════════════
    # 6. POSITION CAP SWEEP
    # ═══════════════════════════════════════════════════════
    print(f"{'=' * 95}")
    print(f"  6. POSITION CAP SWEEP (N=12, 15d)")
    print("=" * 95)
    print(HEADER)
    print(DIV)

    for cap in [0.10, 0.12, 0.15, 0.20, 0.25, 1.0]:
        r = run_backtest(daily, signal_df=best_sig, vix_series=vix,
                         top_n=12, rebal_freq=15, use_vol_scaling=False,
                         max_pos_pct=cap,
                         label=f"Cap={cap:.0%}" if cap < 1.0 else "No cap")
        if r:
            print(row(r))

    # ═══════════════════════════════════════════════════════
    # 7. FINAL COMBINED: BEST CONFIG + FRAGILITY
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 95}")
    print(f"  7. FINAL COMBINED RESULTS")
    print("=" * 95)
    print(HEADER)
    print(DIV)

    # Run the absolute best config
    final = run_backtest(daily, signal_df=best_sig, vix_series=vix,
                         top_n=12, rebal_freq=15, use_vol_scaling=False,
                         label="FINAL: Best combined")
    if final:
        print(row(final))
        print(f"\n  Fragility: {fragility(final, [0, 5, 10, 15, 20])}")

    # Compare production V2
    v2 = run_backtest(daily, signal_df=mom.rank(axis=1, pct=True), vix_series=vix,
                      top_n=10, rebal_freq=10, use_vol_scaling=True,
                      label="V2 Production (baseline)")
    if v2:
        print(f"\n  V2 baseline: {row(v2)}")
        print(f"  V2 fragility: {fragility(v2, [0, 5, 10])}")

    # ═══════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 95}")
    print("  FINAL VERDICT")
    print("=" * 95)
    if final:
        print(f"""
  Best Configuration:
    Signal: {best_w_label} (blended momentum + residual momentum)
    Residual lookback: {best_resid_lb}d
    Stocks: 12
    Rebalance: 15 trading days
    Vol scaling: OFF
    VIX filter: 25/35 (reduce/flat)
    Position cap: 15%
    Cost: 25 bps

  Results:
    CAGR: {final['cagr']:+.1f}%  (SPY: {final['spy_cagr']:+.1f}%)
    Alpha: {final['alpha']:+.1f}%
    Sharpe: {final['sharpe']:.3f}
    Sortino: {final['sortino']:.3f}
    Max DD: {final['max_dd']:.1f}%
    PF: {final['pf']:.3f}
    Total trades: {final['trades']}

  vs V2:
    Sharpe: {v2['sharpe']:.3f} → {final['sharpe']:.3f} ({(final['sharpe']/v2['sharpe']-1)*100:+.0f}%)
    Max DD: {v2['max_dd']:.1f}% → {final['max_dd']:.1f}% ({(final['max_dd']-v2['max_dd']):.1f}pp better)
    PF: {v2['pf']:.3f} → {final['pf']:.3f} ({(final['pf']/v2['pf']-1)*100:+.1f}%)
    Trades: {v2['trades']} → {final['trades']} ({(final['trades']/v2['trades']-1)*100:+.0f}%)

  PF >= 1.30 target: {'ACHIEVED' if final['pf'] >= 1.30 else 'NOT YET (' + str(round(final['pf'], 3)) + ')'}
""")
    print("=" * 95)
