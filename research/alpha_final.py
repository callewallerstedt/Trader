#!/usr/bin/env python3
"""
Final optimization: combine all findings to build production V3.

Key findings from rounds 1-2:
  - Signal: M=0.5 R=0.3 A=0.2 is best (mom + residual + acceleration)
  - VIX 20/30 pushes PF above 1.30 (Sh 1.48, PF 1.306)
  - N=15/25d or N=12/15d are best N/rebal combos
  - No vol scaling
  - Residual 126d lookback confirmed
  - Position cap doesn't matter much

This round:
  1. Combine VIX 20/30 with N/rebal grid
  2. Analyze VIX filter behavior during crisis periods (full backtest)
  3. Find absolute best config
  4. Run full validation suite
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
    run_backtest, fragility, HEADER, DIV, row,
)


if __name__ == "__main__":
    print("Loading data...")
    daily = load()
    vix = load_vix()
    symbols = sorted(daily["symbol"].unique())
    tradeable = [s for s in symbols if s not in NON_TRADEABLE]
    pivoted = daily.pivot_table(index="timestamp", columns="symbol", values="close").sort_index().ffill()
    print(f"  {len(tradeable)} tradeable, {daily['timestamp'].dt.date.nunique()} days")

    print("\nComputing signals...")
    mom = _blended_momentum(pivoted, tradeable, [20, 60, 126], [1.0, 2.0, 1.0])
    print("  Residual momentum 126d...", flush=True)
    resid = compute_residual_momentum(pivoted, tradeable, lookback=126)
    print("  Momentum acceleration...", flush=True)
    accel = compute_momentum_acceleration(pivoted, tradeable, short_lb=21, long_lb=126)

    mom_rank = mom.rank(axis=1, pct=True)
    resid_rank = resid.rank(axis=1, pct=True)
    accel_rank = accel.rank(axis=1, pct=True)

    # Best signal from round 2
    sig = mom_rank * 0.5 + resid_rank * 0.3 + accel_rank * 0.2
    print("  Done.\n")

    # =====================================================
    # 1. FULL GRID: VIX x N x REBAL
    # =====================================================
    print("=" * 100)
    print("  1. FULL GRID: VIX x N x REBAL")
    print("=" * 100)
    print(f"  {'Config':<58s} {'CAGR':>7s} {'Sh':>5s} {'DD':>7s} {'Sort':>5s} {'PF':>7s} {'Trd':>5s}")
    print(f"  {'-' * 95}")

    all_results = []
    for vr, vf in [(20, 30), (22, 33), (25, 35)]:
        for n in [10, 12, 15]:
            for freq in [15, 20, 25]:
                r = run_backtest(daily, signal_df=sig, vix_series=vix,
                                 top_n=n, rebal_freq=freq, use_vol_scaling=False,
                                 vix_reduce=vr, vix_flat=vf,
                                 label=f"VIX={vr}/{vf} N={n} rebal={freq}d")
                if r:
                    all_results.append(r)

    all_results.sort(key=lambda x: x["pf"], reverse=True)
    for r in all_results[:20]:
        print(row(r))

    print(f"\n  Top 5 by Sharpe/DD ratio:")
    all_results.sort(key=lambda x: x["sharpe"] / abs(x["max_dd"]) if x["max_dd"] != 0 else 0, reverse=True)
    for r in all_results[:5]:
        ratio = r["sharpe"] / abs(r["max_dd"]) * 100
        print(f"  {row(r)}  Sh/DD={ratio:.1f}")

    # =====================================================
    # 2. VIX REBOUND ANALYSIS
    # =====================================================
    print(f"\n{'=' * 100}")
    print("  2. VIX FILTER REBOUND ANALYSIS")
    print("    Does tighter VIX filter miss rebounds?")
    print("=" * 100)

    vix_dates = vix.index
    dates = pivoted.index

    # Find periods where VIX was elevated
    print("\n  VIX spike episodes and strategy behavior:")
    episodes = [
        ("COVID Mar 2020",    "2020-02-20", "2020-04-30"),
        ("COVID rebound",     "2020-03-23", "2020-06-30"),
        ("VIX spike Oct 2018","2018-10-01", "2018-12-31"),
        ("VIX rebound Jan 19","2019-01-01", "2019-03-31"),
        ("2022 selloff",      "2022-01-01", "2022-06-30"),
        ("2022 Oct rebound",  "2022-10-01", "2023-03-31"),
        ("Aug 2024 VIX spike","2024-07-15", "2024-09-30"),
    ]

    for name, start, end in episodes:
        mask = (pivoted.index >= start) & (pivoted.index <= end)
        if mask.sum() < 5:
            continue
        sub_piv = pivoted.loc[mask]
        if "SPY" not in sub_piv.columns or len(sub_piv) < 5:
            continue
        spy_ret = (sub_piv["SPY"].iloc[-1] / sub_piv["SPY"].iloc[0] - 1) * 100

        # VIX during this period
        vix_mask = (vix.index >= start) & (vix.index <= end)
        vix_sub = vix[vix_mask]
        vix_avg = vix_sub.mean() if len(vix_sub) > 0 else 0
        vix_max = vix_sub.max() if len(vix_sub) > 0 else 0

        # What would each VIX filter do?
        print(f"\n  {name} ({start} to {end})")
        print(f"    SPY: {spy_ret:+.1f}%  VIX avg: {vix_avg:.1f}  VIX peak: {vix_max:.1f}")

        for vr, vf in [(20, 30), (25, 35), (999, 999)]:
            days_reduced = 0
            days_flat = 0
            cfg = Config(vix_reduce_threshold=vr, vix_flat_threshold=vf)
            for dt in sub_piv.index:
                ts = pd.Timestamp(dt)
                v = vix[ts] if ts in vix.index else None
                if v is not None:
                    mult = _vix_exposure_multiplier(v, cfg)
                    if mult <= 0:
                        days_flat += 1
                    elif mult < 1.0:
                        days_reduced += 1
            total = len(sub_piv)
            label = f"VIX {vr}/{vf}" if vr < 999 else "No filter"
            print(f"    {label:>14s}: {days_reduced:>3d} reduced + {days_flat:>3d} flat = {days_reduced+days_flat:>3d}/{total} days affected ({(days_reduced+days_flat)/total*100:.0f}%)")

    # =====================================================
    # 3. SUBPERIOD VALIDATION ON BEST CONFIG
    # =====================================================
    print(f"\n{'=' * 100}")
    print("  3. SUBPERIOD VALIDATION (top configs)")
    print("=" * 100)

    configs = [
        ("VIX=20/30 N=15 25d", 15, 25, 20, 30),
        ("VIX=20/30 N=12 15d", 12, 15, 20, 30),
        ("VIX=25/35 N=12 15d", 12, 15, 25, 35),
    ]

    periods = [
        ("2010-2013", "2010-01-01", "2013-12-31"),
        ("2014-2017", "2014-01-01", "2017-12-31"),
        ("2018-2021", "2018-01-01", "2021-12-31"),
        ("2022-2025", "2022-01-01", "2025-12-31"),
    ]

    for cfg_name, n, freq, vr, vf in configs:
        print(f"\n  {cfg_name}:")
        print(f"  {'Period':<15s} {'CAGR':>7s} {'Sh':>5s} {'DD':>7s} {'PF':>7s}")
        print(f"  {'-'*40}")
        for pname, s, e in periods:
            sub = daily[(daily["timestamp"] >= s) & (daily["timestamp"] <= e)]
            sub_piv = sub.pivot_table(index="timestamp", columns="symbol", values="close").sort_index().ffill()
            sub_trade = [t for t in tradeable if t in sub_piv.columns]
            sub_mom = _blended_momentum(sub_piv, sub_trade, [20, 60, 126], [1.0, 2.0, 1.0])
            sub_resid = compute_residual_momentum(sub_piv, sub_trade, lookback=126)
            sub_accel = compute_momentum_acceleration(sub_piv, sub_trade, short_lb=21, long_lb=126)
            if sub_resid is not None:
                sub_sig = sub_mom.rank(axis=1, pct=True) * 0.5 + sub_resid.rank(axis=1, pct=True) * 0.3 + sub_accel.rank(axis=1, pct=True) * 0.2
                r = run_backtest(sub, signal_df=sub_sig, vix_series=vix,
                                 top_n=n, rebal_freq=freq, use_vol_scaling=False,
                                 vix_reduce=vr, vix_flat=vf,
                                 label=pname)
                if r:
                    print(f"  {pname:<15s} {r['cagr']:>+6.1f}% {r['sharpe']:>5.2f} {r['max_dd']:>6.1f}% {r['pf']:>6.3f}")
                else:
                    print(f"  {pname:<15s} insufficient data")
            else:
                print(f"  {pname:<15s} residual failed")

    # =====================================================
    # 4. COST SENSITIVITY ON TOP CONFIGS
    # =====================================================
    print(f"\n{'=' * 100}")
    print("  4. COST SENSITIVITY ON TOP CONFIGS")
    print("=" * 100)

    for cfg_name, n, freq, vr, vf in configs[:2]:
        print(f"\n  {cfg_name}:")
        print(f"  {'Cost':>8s} {'CAGR':>7s} {'Sh':>5s} {'DD':>7s} {'PF':>7s}")
        print(f"  {'-'*35}")
        for bps in [0, 10, 25, 50, 75]:
            r = run_backtest(daily, signal_df=sig, vix_series=vix,
                             top_n=n, rebal_freq=freq, use_vol_scaling=False,
                             vix_reduce=vr, vix_flat=vf, cost_bps=bps,
                             label=f"{bps}bps")
            if r:
                print(f"  {bps:>6d}bp {r['cagr']:>+6.1f}% {r['sharpe']:>5.2f} {r['max_dd']:>6.1f}% {r['pf']:>6.3f}")

    # =====================================================
    # 5. FRAGILITY TEST ON TOP 3
    # =====================================================
    print(f"\n{'=' * 100}")
    print("  5. FRAGILITY TEST")
    print("=" * 100)

    for cfg_name, n, freq, vr, vf in configs:
        r = run_backtest(daily, signal_df=sig, vix_series=vix,
                         top_n=n, rebal_freq=freq, use_vol_scaling=False,
                         vix_reduce=vr, vix_flat=vf,
                         label=cfg_name)
        if r:
            print(f"\n  {cfg_name}: {fragility(r, [0, 5, 10, 15, 20])}")

    # =====================================================
    # 6. MONTE CARLO
    # =====================================================
    print(f"\n{'=' * 100}")
    print("  6. MONTE CARLO (1000 trials, random subset of days)")
    print("=" * 100)

    r_full = run_backtest(daily, signal_df=sig, vix_series=vix,
                          top_n=15, rebal_freq=25, use_vol_scaling=False,
                          vix_reduce=20, vix_flat=30,
                          label="Full")
    if r_full:
        d_ret = r_full["daily_returns"]
        n_days = len(d_ret)
        mc_sharpes = []
        mc_pfs = []
        for _ in range(1000):
            idx = np.random.choice(n_days, size=n_days, replace=True)
            sample = d_ret[idx]
            std = np.std(sample)
            sh = float(np.mean(sample) / std * math.sqrt(252)) if std > 1e-12 else 0
            pos = sample[sample > 0]
            neg = sample[sample < 0]
            pf = float(np.sum(pos) / abs(np.sum(neg))) if np.sum(neg) != 0 else 0
            mc_sharpes.append(sh)
            mc_pfs.append(pf)

        mc_sharpes = np.array(mc_sharpes)
        mc_pfs = np.array(mc_pfs)
        print(f"\n  Sharpe: median={np.median(mc_sharpes):.2f}  5th={np.percentile(mc_sharpes,5):.2f}  95th={np.percentile(mc_sharpes,95):.2f}")
        print(f"  PF:     median={np.median(mc_pfs):.3f}  5th={np.percentile(mc_pfs,5):.3f}  95th={np.percentile(mc_pfs,95):.3f}")
        print(f"  P(Sharpe>0): {(mc_sharpes>0).mean()*100:.1f}%")
        print(f"  P(PF>1.0):   {(mc_pfs>1.0).mean()*100:.1f}%")
        print(f"  P(PF>1.15):  {(mc_pfs>1.15).mean()*100:.1f}%")
        print(f"  P(PF>1.30):  {(mc_pfs>1.30).mean()*100:.1f}%")

    # =====================================================
    # FINAL RECOMMENDATION
    # =====================================================
    print(f"\n{'=' * 100}")
    print("  PRODUCTION V3 RECOMMENDATION")
    print("=" * 100)

    # Run the two top contenders
    r_a = run_backtest(daily, signal_df=sig, vix_series=vix,
                       top_n=15, rebal_freq=25, use_vol_scaling=False,
                       vix_reduce=20, vix_flat=30,
                       label="A: N=15 25d VIX=20/30")
    r_b = run_backtest(daily, signal_df=sig, vix_series=vix,
                       top_n=12, rebal_freq=15, use_vol_scaling=False,
                       vix_reduce=20, vix_flat=30,
                       label="B: N=12 15d VIX=20/30")
    r_v2 = run_backtest(daily, signal_df=mom.rank(axis=1, pct=True), vix_series=vix,
                        top_n=10, rebal_freq=10, use_vol_scaling=True,
                        label="V2 baseline")

    print(f"\n  {'Config':<40s} {'CAGR':>7s} {'Sh':>6s} {'DD':>7s} {'Sort':>6s} {'PF':>7s} {'Trd':>5s} {'Alpha':>7s}")
    print(f"  {'-'*85}")
    for r in [r_v2, r_a, r_b]:
        if r:
            print(f"  {r['label']:<40s} {r['cagr']:>+6.1f}% {r['sharpe']:>6.3f} {r['max_dd']:>6.1f}% {r['sortino']:>6.3f} {r['pf']:>6.3f} {r['trades']:>5d} {r['alpha']:>+6.1f}%")

    # Pick winner
    winner = r_a if (r_a and r_a["pf"] >= r_b["pf"]) else r_b
    print(f"""
  WINNER: {winner['label']}

  Signal formula:
    score = 0.5 * momentum_rank + 0.3 * residual_momentum_rank + 0.2 * acceleration_rank

  Where:
    momentum        = blended 20/60/126d returns (weights 1:2:1)
    residual_mom    = stock return - beta * SPY return (126d rolling)
    acceleration    = 21d momentum / |126d momentum| (clipped -3 to 3)

  Parameters:
    Stocks:          {winner['label'].split('N=')[1].split()[0]}
    Rebalance:       {winner['label'].split('d')[0].split()[-1]}d
    VIX filter:      reduce at 20, flat at 30
    Vol scaling:     OFF
    Position cap:    15%
    Cost assumed:    25 bps

  PF target >= 1.30: {'ACHIEVED' if winner['pf'] >= 1.30 else 'NOT MET'}
  PF: {winner['pf']:.3f}
  Sharpe: {winner['sharpe']:.3f}
  Max DD: {winner['max_dd']:.1f}%
  CAGR: {winner['cagr']:+.1f}%
  Alpha: {winner['alpha']:+.1f}%
""")
    print("=" * 100)
