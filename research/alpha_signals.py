#!/usr/bin/env python3
"""
Build stronger selection signals to push PF above 1.30.

Signals implemented:
  A. Residual momentum   - momentum minus market beta exposure
  B. Momentum acceleration - is momentum increasing? (short/long ratio)
  C. Earnings drift proxy - return around detected earnings gaps
  D. Combined ranking    - unified multi-signal score

Also applies:
  - No vol scaling (data shows it hurts)
  - 15d rebalance (reduces turnover, improves PF)
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


# ─────────────────────────────────────────────────────────
# SIGNAL BUILDERS
# ─────────────────────────────────────────────────────────

def compute_residual_momentum(pivoted, tradeable, lookback=126):
    """
    Residual momentum: stock return minus beta * market return.
    Removes market/beta exposure, isolates stock-specific momentum.
    Academic source: Blitz, Huij, Martens (2011).
    """
    if "SPY" not in pivoted.columns:
        return None
    spy_ret = pivoted["SPY"].pct_change()
    residual = pd.DataFrame(np.nan, index=pivoted.index, columns=tradeable)

    for sym in tradeable:
        if sym not in pivoted.columns:
            continue
        stock_ret = pivoted[sym].pct_change()
        for i in range(lookback, len(pivoted)):
            sr = stock_ret.iloc[i - lookback:i].values
            mr = spy_ret.iloc[i - lookback:i].values
            valid = ~(np.isnan(sr) | np.isnan(mr))
            if valid.sum() < lookback * 0.7:
                continue
            slope, intercept, _, _, _ = stats.linregress(mr[valid], sr[valid])
            # Residual = cumulative stock-specific return over period
            resid_rets = sr[valid] - slope * mr[valid]
            residual.iat[i, residual.columns.get_loc(sym)] = float(np.sum(resid_rets))
    return residual


def compute_momentum_acceleration(pivoted, tradeable, short_lb=21, long_lb=126):
    """
    Momentum acceleration: ratio of short-term to long-term momentum.
    Values > 1 = momentum is increasing (early trend phase).
    """
    short_mom = pivoted[tradeable].pct_change(short_lb)
    long_mom = pivoted[tradeable].pct_change(long_lb)
    # Avoid division by zero/negative
    long_abs = long_mom.abs().replace(0, np.nan)
    # Use sign-aware ratio: short * sign(long) / |long|
    accel = short_mom / long_abs
    # Clip extremes
    return accel.clip(-3, 3)


def compute_earnings_drift(pivoted, tradeable, gap_threshold=0.03, drift_window=20):
    """
    Earnings drift proxy: detect large single-day gaps (likely earnings),
    then measure sign of most recent gap as forward drift signal.
    Positive gap = bullish drift, negative gap = bearish.
    """
    drift = pd.DataFrame(0.0, index=pivoted.index, columns=tradeable)
    for sym in tradeable:
        if sym not in pivoted.columns:
            continue
        daily_ret = pivoted[sym].pct_change()
        for i in range(drift_window, len(pivoted)):
            # Look for large gaps in recent window
            window = daily_ret.iloc[i - drift_window:i].values
            valid = window[~np.isnan(window)]
            big_moves = valid[np.abs(valid) > gap_threshold]
            if len(big_moves) > 0:
                # Use the most recent big move as signal
                drift.iat[i, drift.columns.get_loc(sym)] = float(np.sign(big_moves[-1]) * abs(big_moves[-1]))
    return drift


def compute_combined_score(pivoted, tradeable, weights_dict,
                           mom_lookbacks=(20, 60, 126), mom_weights=(1.0, 2.0, 1.0)):
    """
    Build unified ranking from multiple signals.
    weights_dict: e.g. {"momentum": 0.4, "residual": 0.3, "acceleration": 0.2, "drift": 0.1}
    Returns DataFrame with combined z-scores.
    """
    signals = {}

    if "momentum" in weights_dict:
        signals["momentum"] = _blended_momentum(pivoted, tradeable, list(mom_lookbacks), list(mom_weights))

    if "residual" in weights_dict:
        signals["residual"] = compute_residual_momentum(pivoted, tradeable, lookback=126)

    if "acceleration" in weights_dict:
        signals["acceleration"] = compute_momentum_acceleration(pivoted, tradeable, short_lb=21, long_lb=126)

    if "drift" in weights_dict:
        signals["drift"] = compute_earnings_drift(pivoted, tradeable)

    # Cross-sectional z-score each signal, then combine
    combined = pd.DataFrame(0.0, index=pivoted.index, columns=tradeable)
    for name, sig in signals.items():
        if sig is None:
            continue
        w = weights_dict[name]
        # Cross-sectional rank (percentile) at each time step
        ranked = sig.rank(axis=1, pct=True)
        combined += ranked * w

    return combined


# ─────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────

def run_backtest(daily, vix_series=None, signal_df=None,
                 top_n=10, inv_vol=True, max_pos_pct=0.15,
                 rebal_freq=15, trend_sma=200, cost_bps=25,
                 vix_reduce=25, vix_flat=35,
                 use_vol_scaling=False, vol_target=0.15,
                 exposure_min=0.5, exposure_max=1.3,
                 initial_equity=100000.0,
                 label=""):
    """Backtest using pre-computed signal DataFrame for stock ranking."""
    symbols = sorted(daily["symbol"].unique())
    tradeable = [s for s in symbols if s not in NON_TRADEABLE]
    pivoted = daily.pivot_table(index="timestamp", columns="symbol", values="close").sort_index().ffill()
    dates = pivoted.index.tolist()
    n = len(dates)
    if "SPY" not in pivoted.columns:
        return None

    avail = [s for s in tradeable if s in pivoted.columns and s in signal_df.columns]
    if len(avail) < top_n:
        return None

    spy = pivoted["SPY"].values
    spy_sma_arr = pd.Series(spy).rolling(trend_sma, min_periods=trend_sma).mean().values

    stock_vol = {}
    if inv_vol:
        for s in avail:
            stock_vol[s] = pivoted[s].pct_change().rolling(20, min_periods=10).std().values * math.sqrt(252)

    port_ret = pivoted[avail].pct_change().mean(axis=1)
    rvol = port_ret.rolling(20, min_periods=10).std().values * math.sqrt(252)

    vix_dict = {}
    if vix_series is not None:
        for dt in dates:
            ts = pd.Timestamp(dt)
            if ts in vix_series.index:
                vix_dict[dt] = float(vix_series[ts])

    warmup = max(200, trend_sma, 130) + 10
    equity = initial_equity
    holdings = []
    weights = {}
    equity_curve = np.full(n, np.nan)
    spy_norm = np.full(n, np.nan)
    spy_base = spy[warmup] if warmup < n else spy[0]
    total_trades = 0
    last_rebalance = 0
    trend_strength = 1.0
    cost_frac = cost_bps / 10000

    for i in range(n):
        spy_norm[i] = initial_equity * (spy[i] / spy_base) if spy_base > 0 else initial_equity
        if i < warmup:
            equity_curve[i] = equity
            continue

        if use_vol_scaling:
            rv = rvol[i-1] if i-1 < len(rvol) and not np.isnan(rvol[i-1]) and rvol[i-1] > 0 else vol_target
            exposure = float(np.clip(vol_target / rv, exposure_min, exposure_max))
        else:
            exposure = 1.0

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
            if np.isnan(spy_sma_arr[i]):
                trend_strength = 0.0
            else:
                trend_strength = _trend_signal_gradual(spy[i], spy_sma_arr[i])

            if trend_strength <= 0 or vix_mult <= 0:
                target_h = []
                target_w = {}
            else:
                sig_today = signal_df.iloc[i].dropna() if i < len(signal_df) else pd.Series(dtype=float)
                sig_today = sig_today[sig_today > 0]

                if len(sig_today) >= top_n:
                    ranked = sig_today.sort_values(ascending=False)
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
                    cost = turnover * cost_frac * equity / max(top_n * 2, 1)
                    equity -= cost
                    total_trades += turnover
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

    return {
        "label": label, "cagr": round(cagr * 100, 2),
        "spy_cagr": round(spy_cagr * 100, 2),
        "alpha": round((cagr - spy_cagr) * 100, 2),
        "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
        "max_dd": round(max_dd * 100, 2),
        "pf": round(pf, 3),
        "trades": total_trades, "years": round(years, 1),
        "ann_vol": round(std * math.sqrt(252) * 100, 2),
        "daily_returns": d_ret,
    }


def fragility(result, pcts=(0, 5, 10)):
    if result is None:
        return ""
    d = result["daily_returns"]
    parts = []
    for p in pcts:
        deg = np.where(d > 0, d * (1 - p/100), d * (1 + p/100))
        std = np.std(deg)
        sh = float(np.mean(deg) / std * math.sqrt(252)) if std > 1e-12 else 0
        pf = float(np.sum(deg[deg > 0]) / abs(np.sum(deg[deg < 0]))) if np.sum(deg[deg < 0]) != 0 else 0
        parts.append(f"{p}%:Sh={sh:.2f}/PF={pf:.2f}")
    return " | ".join(parts)


HEADER = f"  {'Config':<50s} {'CAGR':>7s} {'Sh':>5s} {'DD':>7s} {'Sort':>5s} {'PF':>7s} {'Trd':>5s}"
DIV = f"  {'-'*90}"


def row(r):
    if r is None:
        return f"  {'INSUFFICIENT DATA':<50s}"
    return (f"  {r['label']:<50s} {r['cagr']:>+6.1f}% {r['sharpe']:>5.2f} {r['max_dd']:>6.1f}% "
            f"{r['sortino']:>5.2f} {r['pf']:>6.3f} {r['trades']:>5d}")


if __name__ == "__main__":
    print("Loading data...")
    daily = load()
    vix = load_vix()
    symbols = sorted(daily["symbol"].unique())
    tradeable = [s for s in symbols if s not in NON_TRADEABLE]
    pivoted = daily.pivot_table(index="timestamp", columns="symbol", values="close").sort_index().ffill()
    print(f"  {len(tradeable)} tradeable, {daily['timestamp'].dt.date.nunique()} days\n")

    # ═══════════════════════════════════════════════════════
    # COMPUTE ALL SIGNALS
    # ═══════════════════════════════════════════════════════
    print("Computing signals...")
    print("  Blended momentum...", flush=True)
    mom = _blended_momentum(pivoted, tradeable, [20, 60, 126], [1.0, 2.0, 1.0])

    print("  Residual momentum (126d)...", flush=True)
    resid = compute_residual_momentum(pivoted, tradeable, lookback=126)

    print("  Momentum acceleration...", flush=True)
    accel = compute_momentum_acceleration(pivoted, tradeable, short_lb=21, long_lb=126)

    print("  Earnings drift proxy...", flush=True)
    drift = compute_earnings_drift(pivoted, tradeable)

    print("  Done.\n")

    # ═══════════════════════════════════════════════════════
    # 1. INDIVIDUAL SIGNALS
    # ═══════════════════════════════════════════════════════
    COMMON = dict(vix_series=vix, top_n=10, rebal_freq=15, use_vol_scaling=False)

    print("=" * 95)
    print("  1. INDIVIDUAL SIGNALS (15d rebal, no vol scaling)")
    print("=" * 95)
    print(HEADER)
    print(DIV)

    r_mom = run_backtest(daily, signal_df=mom, label="A. Raw momentum", **COMMON)
    print(row(r_mom))

    r_resid = run_backtest(daily, signal_df=resid, label="B. Residual momentum (vs SPY)", **COMMON)
    print(row(r_resid))

    r_accel = run_backtest(daily, signal_df=accel, label="C. Momentum acceleration (21/126d)", **COMMON)
    print(row(r_accel))

    r_drift = run_backtest(daily, signal_df=drift, label="D. Earnings drift proxy", **COMMON)
    print(row(r_drift))

    # ═══════════════════════════════════════════════════════
    # 2. COMBINED RANKINGS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 95}")
    print("  2. COMBINED RANKING MODELS")
    print("=" * 95)
    print(HEADER)
    print(DIV)

    combos = [
        ("0.6 mom + 0.4 residual",
         {"momentum": 0.6, "residual": 0.4}),
        ("0.5 mom + 0.3 residual + 0.2 accel",
         {"momentum": 0.5, "residual": 0.3, "acceleration": 0.2}),
        ("0.4 mom + 0.3 residual + 0.2 accel + 0.1 drift",
         {"momentum": 0.4, "residual": 0.3, "acceleration": 0.2, "drift": 0.1}),
        ("0.5 mom + 0.5 residual",
         {"momentum": 0.5, "residual": 0.5}),
        ("0.4 residual + 0.4 mom + 0.2 drift",
         {"momentum": 0.4, "residual": 0.4, "drift": 0.2}),
        ("0.7 residual + 0.3 accel",
         {"residual": 0.7, "acceleration": 0.3}),
        ("0.5 residual + 0.3 mom + 0.2 accel",
         {"residual": 0.5, "momentum": 0.3, "acceleration": 0.2}),
        ("0.3 mom + 0.3 residual + 0.2 accel + 0.2 drift",
         {"momentum": 0.3, "residual": 0.3, "acceleration": 0.2, "drift": 0.2}),
    ]

    best_pf = 0
    best_label = ""
    best_weights = None
    for label, w in combos:
        sig = compute_combined_score(pivoted, tradeable, w)
        r = run_backtest(daily, signal_df=sig, label=label, **COMMON)
        if r:
            print(row(r))
            if r["pf"] > best_pf:
                best_pf = r["pf"]
                best_label = label
                best_weights = w

    # ═══════════════════════════════════════════════════════
    # 3. BEST COMBO: PARAMETER SWEEP
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 95}")
    print(f"  3. BEST COMBO ({best_label}) - PARAMETER SWEEP")
    print("=" * 95)
    print(HEADER)
    print(DIV)

    best_sig = compute_combined_score(pivoted, tradeable, best_weights)

    for n_stocks in [7, 10, 12]:
        for freq in [10, 15, 20]:
            r = run_backtest(daily, signal_df=best_sig, top_n=n_stocks, rebal_freq=freq,
                             vix_series=vix, use_vol_scaling=False,
                             label=f"N={n_stocks} rebal={freq}d")
            if r:
                print(row(r))

    # ═══════════════════════════════════════════════════════
    # 4. COST SENSITIVITY ON BEST
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 95}")
    print(f"  4. COST SENSITIVITY (best combo)")
    print("=" * 95)
    print(HEADER)
    print(DIV)

    for bps in [0, 10, 25, 50, 75]:
        r = run_backtest(daily, signal_df=best_sig, cost_bps=bps,
                         vix_series=vix, top_n=10, rebal_freq=15, use_vol_scaling=False,
                         label=f"Cost={bps}bps")
        if r:
            print(row(r))

    # ═══════════════════════════════════════════════════════
    # 5. SUBPERIOD VALIDATION
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 95}")
    print(f"  5. SUBPERIOD VALIDATION (best combo)")
    print("=" * 95)
    print(HEADER)
    print(DIV)

    periods = [
        ("2010-2013", "2010-01-01", "2013-12-31"),
        ("2014-2017", "2014-01-01", "2017-12-31"),
        ("2018-2021", "2018-01-01", "2021-12-31"),
        ("2022-2025", "2022-01-01", "2025-12-31"),
        ("Full",      "2009-01-01", "2026-12-31"),
    ]
    for name, s, e in periods:
        sub = daily[(daily["timestamp"] >= s) & (daily["timestamp"] <= e)]
        sub_piv = sub.pivot_table(index="timestamp", columns="symbol", values="close").sort_index().ffill()
        sub_trade = [s for s in tradeable if s in sub_piv.columns]
        sub_sig = compute_combined_score(sub_piv, sub_trade, best_weights)
        r = run_backtest(sub, signal_df=sub_sig, vix_series=vix,
                         top_n=10, rebal_freq=15, use_vol_scaling=False,
                         label=name)
        if r:
            print(row(r))
        else:
            print(f"  {name:<50s} insufficient data")

    # ═══════════════════════════════════════════════════════
    # 6. FRAGILITY ON BEST
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 95}")
    print(f"  6. FRAGILITY TEST (best combo)")
    print("=" * 95)
    r = run_backtest(daily, signal_df=best_sig, vix_series=vix,
                     top_n=10, rebal_freq=15, use_vol_scaling=False,
                     label="Best combo")
    if r:
        print(f"  {fragility(r, [0, 5, 10, 15, 20])}")

    # ═══════════════════════════════════════════════════════
    # 7. HEAD TO HEAD: OLD vs NEW
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 95}")
    print(f"  7. HEAD-TO-HEAD: V2 vs V3")
    print("=" * 95)
    print(HEADER)
    print(DIV)

    v2 = run_backtest(daily, signal_df=mom, vix_series=vix,
                      top_n=10, rebal_freq=10, use_vol_scaling=True,
                      label="V2 (momentum, 10d, vol-scaled)")
    print(row(v2))

    v3 = run_backtest(daily, signal_df=best_sig, vix_series=vix,
                      top_n=10, rebal_freq=15, use_vol_scaling=False,
                      label=f"V3 ({best_label}, 15d, no vol)")
    print(row(v3))

    print(f"\n  V2 fragility: {fragility(v2)}")
    print(f"  V3 fragility: {fragility(v3)}")

    # ═══════════════════════════════════════════════════════
    # FINAL RECOMMENDATION
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 95}")
    print("  FINAL RECOMMENDATION")
    print("=" * 95)
    print(f"\n  Best signal combination: {best_label}")
    print(f"  Signal weights: {best_weights}")
    if v3:
        print(f"  PF: {v3['pf']:.3f}  Sharpe: {v3['sharpe']:.3f}  DD: {v3['max_dd']:.1f}%  CAGR: {v3['cagr']:+.1f}%")
        pf_target = v3['pf'] >= 1.30
        print(f"  PF >= 1.30 target: {'ACHIEVED' if pf_target else 'NOT YET'}")
    print("=" * 95)
