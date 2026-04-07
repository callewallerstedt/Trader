#!/usr/bin/env python3
"""Detailed analysis of the production strategy with full metrics."""
from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strategy.data import load
from research.strategy_research import backtest_strategy, StrategyConfig

def annual_returns_table(daily: pd.DataFrame, config: StrategyConfig):
    """Compute year-by-year returns for strategy vs SPY."""
    tradeable = [s for s in daily["symbol"].unique() if s not in {"SPY","DIA","IWM","TLT","GLD","EFA","EEM"}]
    pivoted = daily.pivot_table(index="timestamp", columns="symbol", values="close").sort_index().ffill()
    spy = pivoted["SPY"]

    years = sorted(set(spy.index.year))
    print(f"\n{'Year':<6} {'Strategy':>10} {'SPY':>10} {'Alpha':>10} {'Status':>10}")
    print("-" * 50)

    for yr in years:
        yr_data = daily[daily["timestamp"].dt.year <= yr]
        if yr_data["timestamp"].dt.year.nunique() < 2:
            continue
        prev_data = daily[daily["timestamp"].dt.year < yr]
        if prev_data.empty or yr_data.empty:
            continue

        full = backtest_strategy(yr_data, config)
        prev = backtest_strategy(prev_data, config)

        if "error" in full or "error" in prev:
            continue
        if prev["final_equity"] <= 0:
            continue

        strat_yr_ret = full["final_equity"] / prev["final_equity"] - 1

        spy_end = pivoted["SPY"].loc[pivoted["SPY"].index.year <= yr].iloc[-1]
        spy_prev = pivoted["SPY"].loc[pivoted["SPY"].index.year < yr]
        if spy_prev.empty:
            continue
        spy_yr_ret = spy_end / spy_prev.iloc[-1] - 1

        alpha = strat_yr_ret - spy_yr_ret
        status = "WIN" if strat_yr_ret > spy_yr_ret else "LOSE"
        color = "+" if strat_yr_ret > 0 else ""
        print(f"{yr:<6} {color}{strat_yr_ret*100:>8.1f}% {spy_yr_ret*100:>+9.1f}% {alpha*100:>+9.1f}%  {status:>8}")


def main():
    print("Loading 16 years of data...")
    daily = load()
    syms = sorted(daily["symbol"].unique())
    dates = daily["timestamp"].dt.date.nunique()
    print(f"  {len(syms)} symbols, {dates} trading days")
    print(f"  {daily['timestamp'].min().date()} to {daily['timestamp'].max().date()}")

    config = StrategyConfig(
        name="PRODUCTION: Multi-TF Momentum Gradual Trend",
        mom_lookbacks=[20, 60, 126], mom_weights=[1, 2, 1], top_n=5,
        trend_sma=200, trend_type="gradual", absolute_mom_filter=True,
        rebalance_freq=5, round_trip_cost_bps=25.0,
    )

    print(f"\nStrategy: {config.name}")
    print(f"  Momentum: blend {config.mom_lookbacks} (weights {config.mom_weights})")
    print(f"  Positions: top {config.top_n}")
    print(f"  Trend filter: {config.trend_type} SMA-{config.trend_sma}")
    print(f"  Absolute momentum filter: {config.absolute_mom_filter}")
    print(f"  Rebalance: every {config.rebalance_freq} days")
    print(f"  Round-trip cost: {config.round_trip_cost_bps} bps")

    # Full period
    print("\n" + "=" * 60)
    print("  FULL PERIOD RESULTS")
    print("=" * 60)
    r = backtest_strategy(daily, config)
    for k, v in sorted(r.items()):
        if k == "name":
            continue
        print(f"  {k:<25} {v}")

    # Stress test with higher costs
    print("\n" + "=" * 60)
    print("  STRESS TESTS")
    print("=" * 60)
    for cost in [15, 20, 25, 30, 35, 40, 50]:
        cfg = StrategyConfig(**{k: getattr(config, k) for k in config.__dataclass_fields__})
        cfg.round_trip_cost_bps = float(cost)
        cfg.name = f"{cost}bps"
        res = backtest_strategy(daily, cfg)
        print(f"  {cost:>3}bps: CAGR {res['cagr_pct']:>+6.1f}% | Sharpe {res['sharpe']:>5.2f} | MaxDD {res['max_drawdown_pct']:>6.1f}% | Trades {res['total_trades']:>5}")

    # Out-of-sample validation
    print("\n" + "=" * 60)
    print("  OUT-OF-SAMPLE SPLITS")
    print("=" * 60)
    splits = [
        ("2010-2017 (IS)", "2017-01-01", True),
        ("2017-2026 (OOS)", "2017-01-01", False),
        ("2010-2019 (IS)", "2020-01-01", True),
        ("2020-2026 (OOS)", "2020-01-01", False),
        ("2010-2022 (IS)", "2022-01-01", True),
        ("2022-2026 (OOS)", "2022-01-01", False),
    ]
    for label, date, is_before in splits:
        cutoff = pd.Timestamp(date)
        subset = daily[daily["timestamp"] < cutoff] if is_before else daily[daily["timestamp"] >= cutoff]
        if subset.empty:
            continue
        cfg = StrategyConfig(**{k: getattr(config, k) for k in config.__dataclass_fields__})
        cfg.name = label
        res = backtest_strategy(subset, cfg)
        if "error" in res:
            print(f"  {label:<20} ERROR: {res['error']}")
            continue
        print(f"  {label:<20} CAGR {res['cagr_pct']:>+6.1f}% | Sharpe {res['sharpe']:>5.2f} | MaxDD {res['max_drawdown_pct']:>6.1f}% | Alpha {res['alpha_pct']:>+5.1f}%")

    # Compare old vs new
    print("\n" + "=" * 60)
    print("  OLD vs NEW STRATEGY")
    print("=" * 60)
    old = StrategyConfig(
        name="OLD: 20d mom top3 SMA100",
        mom_lookbacks=[20], mom_weights=[1.0], top_n=3, trend_sma=100,
        trend_type="binary", absolute_mom_filter=False, rebalance_freq=1,
        round_trip_cost_bps=20.0,
    )
    old_r = backtest_strategy(daily, old)
    new_r = r
    print(f"  {'Metric':<25} {'OLD':>12} {'NEW':>12} {'Change':>12}")
    print(f"  {'-'*60}")
    for metric in ["cagr_pct", "alpha_pct", "sharpe", "sortino", "calmar", "max_drawdown_pct", "total_trades", "monthly_win_rate", "negative_years", "worst_year_pct", "days_invested_pct"]:
        old_v = old_r.get(metric, "N/A")
        new_v = new_r.get(metric, "N/A")
        if isinstance(old_v, (int, float)) and isinstance(new_v, (int, float)):
            diff = new_v - old_v
            print(f"  {metric:<25} {old_v:>12} {new_v:>12} {diff:>+11.1f}")
        else:
            print(f"  {metric:<25} {old_v:>12} {new_v:>12}")


if __name__ == "__main__":
    main()
