#!/usr/bin/env python3
"""
Phase 2: Refine top candidates with out-of-sample validation and robustness tests.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.strategy_research import *


def split_data(daily: pd.DataFrame, split_date: str = "2020-01-01"):
    """Split data into in-sample and out-of-sample."""
    cutoff = pd.Timestamp(split_date)
    in_sample = daily[daily["timestamp"] < cutoff].copy()
    out_sample = daily[daily["timestamp"] >= cutoff].copy()
    return in_sample, out_sample


def robustness_test(daily: pd.DataFrame, config: StrategyConfig) -> dict:
    """Test strategy with different cost levels and time splits."""
    full = backtest_strategy(daily, config)

    in_sample, out_sample = split_data(daily, "2020-01-01")

    cfg_is = StrategyConfig(**{k: getattr(config, k) for k in config.__dataclass_fields__})
    cfg_is.name = f"[IS 2010-2019] {config.name}"
    is_result = backtest_strategy(in_sample, cfg_is)

    cfg_os = StrategyConfig(**{k: getattr(config, k) for k in config.__dataclass_fields__})
    cfg_os.name = f"[OOS 2020-2026] {config.name}"
    os_result = backtest_strategy(out_sample, cfg_os)

    cfg_high_cost = StrategyConfig(**{k: getattr(config, k) for k in config.__dataclass_fields__})
    cfg_high_cost.round_trip_cost_bps = 35.0
    cfg_high_cost.name = f"[35bps cost] {config.name}"
    hc_result = backtest_strategy(daily, cfg_high_cost)

    return {
        "name": config.name,
        "full": full,
        "in_sample": is_result,
        "out_sample": os_result,
        "high_cost": hc_result,
    }


def print_robustness(results: list[dict]):
    print("\n" + "=" * 150)
    print(f"{'Strategy':<55} {'Period':<15} {'CAGR':>6} {'Alpha':>6} {'Sharpe':>7} {'MaxDD':>7} {'NegYr':>6} {'Trades':>7}")
    print("=" * 150)
    for r in results:
        for period, key in [("FULL", "full"), ("IS 2010-19", "in_sample"), ("OOS 2020-26", "out_sample"), ("HIGH COST", "high_cost")]:
            d = r[key]
            if "error" in d:
                print(f"{r['name']:<55} {period:<15} ERROR: {d['error']}")
                continue
            print(
                f"{r['name']:<55} {period:<15} "
                f"{d['cagr_pct']:>5.1f}% "
                f"{d['alpha_pct']:>5.1f}% "
                f"{d['sharpe']:>7.3f} "
                f"{d['max_drawdown_pct']:>6.1f}% "
                f"{d['negative_years']:>5} "
                f"{d['total_trades']:>7}"
            )
        print("-" * 150)


def main():
    print("Loading data...")
    daily = load()
    print(f"  {len(daily['symbol'].unique())} symbols, {daily['timestamp'].dt.date.nunique()} trading days\n")

    # Top candidates from Phase 1 plus refined variations
    candidates = [
        StrategyConfig(
            name="E: Blend20+60 top3 dualSMA50/200 wkly",
            mom_lookbacks=[20, 60], mom_weights=[1, 1], top_n=3,
            trend_sma=200, trend_sma_fast=50, trend_type="dual_sma",
            rebalance_freq=5, round_trip_cost_bps=25.0,
        ),
        StrategyConfig(
            name="B: Blend20+60+126 top5 abs-filter",
            mom_lookbacks=[20, 60, 126], mom_weights=[1, 1, 1], top_n=5,
            trend_sma=100, absolute_mom_filter=True, rebalance_freq=5,
            round_trip_cost_bps=25.0,
        ),
        StrategyConfig(
            name="C: 12m-1m top3 SMA200 monthly",
            mom_lookbacks=[252], mom_weights=[1], skip_recent=21, top_n=3,
            trend_sma=200, rebalance_freq=21, round_trip_cost_bps=25.0,
        ),
        StrategyConfig(
            name="G: Blend20+60+126 top5 invVol SMA100 abs",
            mom_lookbacks=[20, 60, 126], mom_weights=[1, 1, 2], top_n=5,
            trend_sma=100, weighting="inv_vol", absolute_mom_filter=True,
            rebalance_freq=5, round_trip_cost_bps=25.0,
        ),
        StrategyConfig(
            name="I: 12m-1m top5 invVol dualSMA50/200 mthly",
            mom_lookbacks=[252], mom_weights=[1], skip_recent=21, top_n=5,
            trend_sma=200, trend_sma_fast=50, trend_type="dual_sma",
            weighting="inv_vol", rebalance_freq=21, round_trip_cost_bps=25.0,
        ),
        # New refined candidates combining best elements
        StrategyConfig(
            name="R1: Blend20+60+126 top5 eq dualSMA50/200 wkly abs",
            mom_lookbacks=[20, 60, 126], mom_weights=[1, 1, 1], top_n=5,
            trend_sma=200, trend_sma_fast=50, trend_type="dual_sma",
            absolute_mom_filter=True, rebalance_freq=5, round_trip_cost_bps=25.0,
        ),
        StrategyConfig(
            name="R2: Blend20+60+126 top3 eq SMA200 wkly abs",
            mom_lookbacks=[20, 60, 126], mom_weights=[1, 1, 1], top_n=3,
            trend_sma=200, absolute_mom_filter=True, rebalance_freq=5,
            round_trip_cost_bps=25.0,
        ),
        StrategyConfig(
            name="R3: Blend60+126 top5 eq SMA200 wkly abs",
            mom_lookbacks=[60, 126], mom_weights=[1, 1], top_n=5,
            trend_sma=200, absolute_mom_filter=True, rebalance_freq=5,
            round_trip_cost_bps=25.0,
        ),
        StrategyConfig(
            name="R4: Blend20+60+126 top5 eq gradual200 wkly abs",
            mom_lookbacks=[20, 60, 126], mom_weights=[1, 2, 1], top_n=5,
            trend_sma=200, trend_type="gradual", absolute_mom_filter=True,
            rebalance_freq=5, round_trip_cost_bps=25.0,
        ),
        StrategyConfig(
            name="R5: Blend20+60 top5 eq dualSMA50/200 wkly abs",
            mom_lookbacks=[20, 60], mom_weights=[1, 1], top_n=5,
            trend_sma=200, trend_sma_fast=50, trend_type="dual_sma",
            absolute_mom_filter=True, rebalance_freq=5, round_trip_cost_bps=25.0,
        ),
        StrategyConfig(
            name="R6: Blend20+60+126 top4 eq dualSMA50/200 wkly abs",
            mom_lookbacks=[20, 60, 126], mom_weights=[1, 1, 1], top_n=4,
            trend_sma=200, trend_sma_fast=50, trend_type="dual_sma",
            absolute_mom_filter=True, rebalance_freq=5, round_trip_cost_bps=25.0,
        ),
        StrategyConfig(
            name="R7: Blend20+60+126 top5 eq SMA100 wkly abs 30bps",
            mom_lookbacks=[20, 60, 126], mom_weights=[1, 1, 1], top_n=5,
            trend_sma=100, absolute_mom_filter=True, rebalance_freq=5,
            round_trip_cost_bps=30.0,
        ),
    ]

    results = []
    for cfg in candidates:
        print(f"Robustness testing: {cfg.name}...")
        results.append(robustness_test(daily, cfg))

    print_robustness(results)

    # Final ranking
    print("\n\n" + "=" * 100)
    print("FINAL RANKING (sorted by minimum of IS/OOS Sharpe for robustness)")
    print("=" * 100)
    ranked = []
    for r in results:
        is_s = r["in_sample"].get("sharpe", 0) if "error" not in r["in_sample"] else 0
        os_s = r["out_sample"].get("sharpe", 0) if "error" not in r["out_sample"] else 0
        full_s = r["full"].get("sharpe", 0) if "error" not in r["full"] else 0
        min_s = min(is_s, os_s) if os_s > 0 else is_s
        ranked.append((r["name"], min_s, full_s, is_s, os_s, r))
    ranked.sort(key=lambda x: x[1], reverse=True)

    for i, (name, min_s, full_s, is_s, os_s, r) in enumerate(ranked):
        f = r["full"]
        print(f"\n  {i+1}. {name}")
        print(f"     Full:  CAGR {f.get('cagr_pct',0):+.1f}% | Alpha {f.get('alpha_pct',0):+.1f}% | Sharpe {full_s:.3f} | MaxDD {f.get('max_drawdown_pct',0):.1f}%")
        print(f"     IS:    Sharpe {is_s:.3f} | OOS: Sharpe {os_s:.3f} | Min: {min_s:.3f}")


if __name__ == "__main__":
    main()
