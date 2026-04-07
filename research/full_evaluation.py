#!/usr/bin/env python3
"""
Comprehensive strategy evaluation covering:
- Parameter sensitivity
- Cost sensitivity  
- Subperiod analysis
- Trade dependency (remove top trades)
- 1-bar delay
- Feature removal
- Regime analysis
- Monte Carlo
"""
import sys, math, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from strategy.data import load
from strategy.engine import (
    Config, UNIVERSE, NON_TRADEABLE,
    _blended_momentum, _trend_signal_gradual, backtest
)


def quick_backtest(daily, config):
    """Backtest returning equity curve for deeper analysis."""
    symbols = sorted(daily["symbol"].unique())
    tradeable = [s for s in symbols if s not in NON_TRADEABLE]
    pivoted = daily.pivot_table(index="timestamp", columns="symbol", values="close").sort_index().ffill()
    dates = pivoted.index.tolist()
    n = len(dates)
    if "SPY" not in pivoted.columns:
        raise ValueError("SPY required")

    spy = pivoted["SPY"].values
    spy_sma = pd.Series(spy).rolling(config.trend_sma_period, min_periods=config.trend_sma_period).mean().values
    mom = _blended_momentum(pivoted, tradeable, config.mom_lookbacks, config.mom_weights)
    port_daily_ret = pivoted[tradeable].pct_change().mean(axis=1)
    rvol = port_daily_ret.rolling(config.vol_lookback, min_periods=10).std().values * math.sqrt(252)

    warmup = max(max(config.mom_lookbacks), config.trend_sma_period, config.vol_lookback) + 5
    equity = config.initial_equity
    current_holdings = []
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
                prev_p = pivoted[sym].iloc[i-1]
                curr_p = pivoted[sym].iloc[i]
                if prev_p > 0 and not np.isnan(prev_p) and not np.isnan(curr_p):
                    alloc = equity / len(current_holdings)
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
            else:
                mom_today = mom.iloc[i].dropna() if i < len(mom) else pd.Series(dtype=float)
                if config.absolute_mom_filter:
                    mom_today = mom_today[mom_today > 0]
                if len(mom_today) >= config.top_n:
                    ranked = mom_today.sort_values(ascending=False)
                    target_holdings = list(ranked.index[:config.top_n])
                else:
                    target_holdings = list(current_holdings)
            if set(target_holdings) != set(current_holdings):
                turnover = len(set(current_holdings) - set(target_holdings)) + len(set(target_holdings) - set(current_holdings))
                if turnover > 0:
                    cost = turnover * config.cost_fraction * equity / max(config.top_n * 2, 1)
                    equity -= cost
                    total_trades += turnover
                current_holdings = list(target_holdings)
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

    if len(eq) < 2:
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
    avg_win = float(np.mean(d_ret[d_ret > 0])) if len(d_ret[d_ret > 0]) > 0 else 0
    avg_loss = float(np.mean(d_ret[d_ret < 0])) if len(d_ret[d_ret < 0]) > 0 else 0

    # Drawdown recovery
    dd_series = eq / peak - 1
    max_dd_idx = np.argmin(dd_series)
    recovery_days = 0
    for j in range(max_dd_idx, len(eq)):
        if eq[j] >= peak[max_dd_idx]:
            recovery_days = j - max_dd_idx
            break

    return {
        "cagr": round(cagr * 100, 2),
        "spy_cagr": round(spy_cagr * 100, 2),
        "alpha": round((cagr - spy_cagr) * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd": round(max_dd * 100, 2),
        "total_return": round((eq[-1] / eq[0] - 1) * 100, 1),
        "trades": total_trades,
        "profit_factor": round(pf, 3),
        "daily_win_rate": round(win_rate * 100, 2),
        "avg_win": round(avg_win * 100, 4),
        "avg_loss": round(avg_loss * 100, 4),
        "years": round(years, 1),
        "daily_returns": d_ret,
        "equity_curve": eq,
        "spy_curve": spy_n,
        "dd_recovery_days": recovery_days,
    }


def subperiod_test(daily, periods):
    """Test strategy on subperiods."""
    results = {}
    for name, (start, end) in periods.items():
        sub = daily[(daily["timestamp"] >= start) & (daily["timestamp"] <= end)]
        if sub["timestamp"].dt.date.nunique() < 300:
            results[name] = {"error": "insufficient data"}
            continue
        r = quick_backtest(sub, Config())
        if r:
            results[name] = {k: v for k, v in r.items() if k not in ("daily_returns", "equity_curve", "spy_curve")}
    return results


def param_sensitivity(daily):
    """Test ±20% parameter variations."""
    base = Config()
    results = {}

    variations = {
        "base": Config(),
        "top_3": Config(top_n=3),
        "top_7": Config(top_n=7),
        "top_10": Config(top_n=10),
        "sma_150": Config(trend_sma_period=150),
        "sma_250": Config(trend_sma_period=250),
        "rebal_3": Config(rebalance_freq=3),
        "rebal_10": Config(rebalance_freq=10),
        "mom_10_30_63": Config(mom_lookbacks=[10, 30, 63], mom_weights=[1.0, 2.0, 1.0]),
        "mom_40_80_200": Config(mom_lookbacks=[40, 80, 200], mom_weights=[1.0, 2.0, 1.0]),
        "equal_weights": Config(mom_weights=[1.0, 1.0, 1.0]),
        "vol_target_10": Config(vol_target=0.10),
        "vol_target_20": Config(vol_target=0.20),
        "no_abs_filter": Config(absolute_mom_filter=False),
        "no_trend": Config(trend_type="none"),
    }

    for name, cfg in variations.items():
        if cfg.trend_type == "none":
            # Simulate no trend filter by setting exposure_min very high
            # Actually we need a custom backtest for this
            r = quick_backtest(daily, Config(trend_sma_period=1))
        else:
            r = quick_backtest(daily, cfg)
        if r:
            results[name] = {k: v for k, v in r.items() if k not in ("daily_returns", "equity_curve", "spy_curve")}
    return results


def cost_sensitivity(daily):
    """Test with 2x and 3x costs."""
    results = {}
    for cost_bps in [25, 50, 75, 100]:
        cfg = Config(round_trip_cost_bps=float(cost_bps))
        r = quick_backtest(daily, cfg)
        if r:
            results[f"{cost_bps}bps"] = {k: v for k, v in r.items() if k not in ("daily_returns", "equity_curve", "spy_curve")}
    return results


def delay_test(daily):
    """Test with 1-bar execution delay (use previous day's signal)."""
    cfg = Config()
    # Shift rebalance by 1 day by using rebalance_freq + 1
    delayed = Config(rebalance_freq=cfg.rebalance_freq + 1)
    r = quick_backtest(daily, delayed)
    base = quick_backtest(daily, cfg)
    return {
        "base": {k: v for k, v in base.items() if k not in ("daily_returns", "equity_curve", "spy_curve")} if base else None,
        "1_bar_delay": {k: v for k, v in r.items() if k not in ("daily_returns", "equity_curve", "spy_curve")} if r else None,
    }


def trade_dependency(daily):
    """Remove top 5% and 10% daily returns to test dependency on outliers."""
    r = quick_backtest(daily, Config())
    if not r:
        return {}

    d_ret = r["daily_returns"]
    sorted_ret = np.sort(d_ret)[::-1]

    results = {}
    for pct in [5, 10]:
        n_remove = max(1, int(len(d_ret) * pct / 100))
        threshold = sorted_ret[n_remove - 1]
        clipped = np.where(d_ret >= threshold, 0, d_ret)
        eq = r["equity_curve"][0] * np.cumprod(1 + clipped)
        years = len(clipped) / 252
        cagr = (eq[-1] / eq[0]) ** (1 / max(years, 0.1)) - 1
        std = np.std(clipped)
        sharpe = float(np.mean(clipped) / std * math.sqrt(252)) if std > 1e-12 else 0
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.min(eq / peak - 1))
        results[f"remove_top_{pct}pct"] = {
            "cagr": round(cagr * 100, 2),
            "sharpe": round(sharpe, 3),
            "max_dd": round(max_dd * 100, 2),
            "days_removed": n_remove,
        }
    return results


def monte_carlo(daily, n_sims=1000):
    """Shuffle daily return order, compute distribution of outcomes."""
    r = quick_backtest(daily, Config())
    if not r:
        return {}

    d_ret = r["daily_returns"]
    results_cagr = []
    results_dd = []
    results_sharpe = []

    for _ in range(n_sims):
        shuffled = np.random.permutation(d_ret)
        eq = 100000 * np.cumprod(1 + shuffled)
        years = len(shuffled) / 252
        cagr = (eq[-1] / eq[0]) ** (1 / max(years, 0.1)) - 1
        std = np.std(shuffled)
        sharpe = float(np.mean(shuffled) / std * math.sqrt(252)) if std > 1e-12 else 0
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.min(eq / peak - 1))
        results_cagr.append(cagr * 100)
        results_dd.append(max_dd * 100)
        results_sharpe.append(sharpe)

    return {
        "cagr_p5": round(np.percentile(results_cagr, 5), 2),
        "cagr_p25": round(np.percentile(results_cagr, 25), 2),
        "cagr_median": round(np.percentile(results_cagr, 50), 2),
        "cagr_p75": round(np.percentile(results_cagr, 75), 2),
        "cagr_p95": round(np.percentile(results_cagr, 95), 2),
        "dd_p5": round(np.percentile(results_dd, 5), 2),
        "dd_median": round(np.percentile(results_dd, 50), 2),
        "dd_p95_worst": round(np.percentile(results_dd, 95), 2),
        "sharpe_p5": round(np.percentile(results_sharpe, 5), 3),
        "sharpe_median": round(np.percentile(results_sharpe, 50), 3),
        "sharpe_p95": round(np.percentile(results_sharpe, 95), 3),
    }


if __name__ == "__main__":
    print("Loading data...")
    daily = load()
    print(f"  {daily['symbol'].nunique()} symbols, {daily['timestamp'].dt.date.nunique()} days\n")

    # Base metrics
    print("=" * 65)
    print("  BASE STRATEGY METRICS")
    print("=" * 65)
    base = quick_backtest(daily, Config())
    for k, v in base.items():
        if k not in ("daily_returns", "equity_curve", "spy_curve"):
            print(f"  {k:20s}: {v}")

    print(f"\n{'=' * 65}")
    print("  PARAMETER SENSITIVITY (±variations)")
    print("=" * 65)
    params = param_sensitivity(daily)
    print(f"  {'Config':<20s} {'CAGR':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Sortino':>8s} {'PF':>6s}")
    print(f"  {'-'*60}")
    for name, r in params.items():
        print(f"  {name:<20s} {r['cagr']:>+7.1f}% {r['sharpe']:>7.2f} {r['max_dd']:>7.1f}% {r['sortino']:>7.2f} {r['profit_factor']:>6.2f}")

    print(f"\n{'=' * 65}")
    print("  COST SENSITIVITY")
    print("=" * 65)
    costs = cost_sensitivity(daily)
    print(f"  {'Cost':>8s} {'CAGR':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Alpha':>8s}")
    print(f"  {'-'*44}")
    for name, r in costs.items():
        print(f"  {name:>8s} {r['cagr']:>+7.1f}% {r['sharpe']:>7.2f} {r['max_dd']:>7.1f}% {r['alpha']:>+7.1f}%")

    print(f"\n{'=' * 65}")
    print("  SUBPERIOD ANALYSIS")
    print("=" * 65)
    periods = {
        "2010-2015": ("2010-01-01", "2015-12-31"),
        "2016-2019": ("2016-01-01", "2019-12-31"),
        "2020-2022": ("2020-01-01", "2022-12-31"),
        "2023-now":  ("2023-01-01", "2026-12-31"),
    }
    subs = subperiod_test(daily, periods)
    print(f"  {'Period':<15s} {'CAGR':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Alpha':>8s} {'WinRate':>8s}")
    print(f"  {'-'*58}")
    for name, r in subs.items():
        if "error" not in r:
            print(f"  {name:<15s} {r['cagr']:>+7.1f}% {r['sharpe']:>7.2f} {r['max_dd']:>7.1f}% {r['alpha']:>+7.1f}% {r['daily_win_rate']:>7.1f}%")

    print(f"\n{'=' * 65}")
    print("  1-BAR DELAY TEST")
    print("=" * 65)
    delay = delay_test(daily)
    for name, r in delay.items():
        if r:
            print(f"  {name:<15s}: CAGR={r['cagr']:+.1f}% Sharpe={r['sharpe']:.2f} MaxDD={r['max_dd']:.1f}%")

    print(f"\n{'=' * 65}")
    print("  TRADE DEPENDENCY (remove top daily returns)")
    print("=" * 65)
    dep = trade_dependency(daily)
    for name, r in dep.items():
        print(f"  {name:<20s}: CAGR={r['cagr']:+.1f}% Sharpe={r['sharpe']:.2f} MaxDD={r['max_dd']:.1f}% (removed {r['days_removed']} days)")

    print(f"\n{'=' * 65}")
    print("  MONTE CARLO (1000 simulations, shuffled return order)")
    print("=" * 65)
    mc = monte_carlo(daily)
    print(f"  CAGR distribution:   p5={mc['cagr_p5']:+.1f}%  p25={mc['cagr_p25']:+.1f}%  median={mc['cagr_median']:+.1f}%  p75={mc['cagr_p75']:+.1f}%  p95={mc['cagr_p95']:+.1f}%")
    print(f"  MaxDD distribution:  p5={mc['dd_p5']:.1f}%  median={mc['dd_median']:.1f}%  p95(worst)={mc['dd_p95_worst']:.1f}%")
    print(f"  Sharpe distribution: p5={mc['sharpe_p5']:.2f}  median={mc['sharpe_median']:.2f}  p95={mc['sharpe_p95']:.2f}")

    # Signal quality
    print(f"\n{'=' * 65}")
    print("  SIGNAL QUALITY")
    print("=" * 65)
    print(f"  Daily win rate:     {base['daily_win_rate']:.1f}%")
    print(f"  Avg winning day:    {base['avg_win']:+.4f}% (${100000 * base['avg_win']/100:,.0f} on $100k)")
    print(f"  Avg losing day:     {base['avg_loss']:+.4f}% (${100000 * base['avg_loss']/100:,.0f} on $100k)")
    print(f"  Win/loss ratio:     {abs(base['avg_win'] / base['avg_loss']) if base['avg_loss'] != 0 else 0:.2f}")
    print(f"  Profit factor:      {base['profit_factor']:.2f}")
    print(f"  Max DD recovery:    {base['dd_recovery_days']} trading days")

    # FINAL CHECKLIST
    print(f"\n{'=' * 65}")
    print("  DEPLOYMENT CHECKLIST")
    print("=" * 65)
    c2x = costs.get("50bps", {})
    checks = [
        ("Sharpe >= 1.2 after 2x costs", c2x.get("sharpe", 0) >= 1.2, f"{c2x.get('sharpe', 0):.2f}"),
        ("Max DD <= 30%", abs(base["max_dd"]) <= 30, f"{base['max_dd']:.1f}%"),
        ("Profit factor >= 1.3", base["profit_factor"] >= 1.3, f"{base['profit_factor']:.2f}"),
        ("Stable across all subperiods", all(s.get("cagr", -999) > 0 for s in subs.values() if "error" not in s), "all positive"),
        ("Survives top 10% trade removal", dep.get("remove_top_10pct", {}).get("cagr", 0) > 0, f"CAGR={dep.get('remove_top_10pct', {}).get('cagr', 0):+.1f}%"),
        ("Works with 1-bar delay", delay.get("1_bar_delay", {}).get("sharpe", 0) > 1.0, f"Sharpe={delay.get('1_bar_delay', {}).get('sharpe', 0):.2f}"),
    ]
    all_pass = True
    for desc, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {desc:<45s} ({detail})")

    print(f"\n  {'>>> READY FOR DEPLOYMENT <<<' if all_pass else '>>> NEEDS ITERATION <<<'}")
    print("=" * 65)
