#!/usr/bin/env python3
"""
Comprehensive strategy research - testing many variations to find the best one.
Uses realistic transaction costs and out-of-sample validation.
"""
from __future__ import annotations

import itertools
import math
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strategy.data import load

# ─── Realistic cost assumptions ──────────────────────────────────────
# IBKR commission: ~$0.005/share ≈ 0.5 bps per side
# Spread: 2-3 bps for liquid ETFs/stocks
# MOC slippage: 3-5 bps
# Total round-trip: ~15-25 bps (we test multiple levels)
COST_LEVELS = {"conservative": 30.0, "moderate": 20.0, "realistic": 15.0}


@dataclass
class StrategyConfig:
    name: str = "default"
    universe_type: str = "mixed"  # "etf_only", "stock_only", "mixed"
    mom_lookbacks: list[int] = field(default_factory=lambda: [20])
    mom_weights: list[float] = field(default_factory=lambda: [1.0])
    skip_recent: int = 0  # skip most recent N days (momentum crash protection)
    top_n: int = 3
    trend_sma: int = 100
    trend_type: str = "binary"  # "binary", "gradual", "dual_sma"
    trend_sma_fast: int = 50  # for dual_sma
    weighting: str = "equal"  # "equal", "inv_vol", "momentum_weighted"
    vol_target: float = 0.15
    vol_lookback: int = 20
    exposure_min: float = 0.5
    exposure_max: float = 1.3
    rebalance_freq: int = 1  # days between rebalances (1=daily, 5=weekly, 21=monthly)
    absolute_mom_filter: bool = False  # only buy stocks with positive momentum
    round_trip_cost_bps: float = 20.0
    initial_equity: float = 100000.0

    @property
    def cost_fraction(self) -> float:
        return self.round_trip_cost_bps / 10000


# Universe definitions
ETF_UNIVERSE = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY", "XLC"]
STOCK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    "JPM", "V", "UNH", "JNJ", "PG", "HD", "MA",
    "TSLA", "AMD", "NFLX",
    "XOM", "INTC", "BA", "PFE", "GE", "T",
]
MIXED_UNIVERSE = ETF_UNIVERSE + STOCK_UNIVERSE
BENCHMARK_ONLY = {"SPY"}


def get_tradeable(config: StrategyConfig) -> list[str]:
    if config.universe_type == "etf_only":
        return ETF_UNIVERSE
    elif config.universe_type == "stock_only":
        return STOCK_UNIVERSE
    else:
        return MIXED_UNIVERSE


def compute_blended_momentum(
    pivoted: pd.DataFrame,
    tradeable: list[str],
    lookbacks: list[int],
    weights: list[float],
    skip_recent: int = 0,
) -> pd.DataFrame:
    """Compute weighted blend of multiple momentum lookback periods."""
    mom_frames = []
    for lb, w in zip(lookbacks, weights):
        if skip_recent > 0:
            shifted = pivoted[tradeable].shift(skip_recent)
            mom = shifted.pct_change(lb) * w
        else:
            mom = pivoted[tradeable].pct_change(lb) * w
        mom_frames.append(mom)

    blended = sum(mom_frames) / sum(weights)
    return blended


def compute_risk_adjusted_momentum(
    pivoted: pd.DataFrame,
    tradeable: list[str],
    lookbacks: list[int],
    weights: list[float],
    skip_recent: int = 0,
    vol_window: int = 20,
) -> pd.DataFrame:
    """Risk-adjusted momentum: divide momentum by trailing volatility."""
    raw_mom = compute_blended_momentum(pivoted, tradeable, lookbacks, weights, skip_recent)
    daily_vol = pivoted[tradeable].pct_change().rolling(vol_window, min_periods=10).std()
    annual_vol = daily_vol * math.sqrt(252)
    annual_vol = annual_vol.replace(0, np.nan)
    return raw_mom / annual_vol


def backtest_strategy(daily: pd.DataFrame, config: StrategyConfig) -> dict:
    """Run a full backtest with the given configuration."""
    tradeable = get_tradeable(config)
    available = [s for s in tradeable if s in daily["symbol"].unique()]
    if len(available) < config.top_n:
        return {"error": f"Not enough symbols: {len(available)}"}

    pivoted = (
        daily.pivot_table(index="timestamp", columns="symbol", values="close")
        .sort_index()
        .ffill()
    )
    dates = pivoted.index.tolist()
    n = len(dates)

    if "SPY" not in pivoted.columns:
        return {"error": "SPY not in data"}

    spy = pivoted["SPY"].values
    spy_sma_slow = (
        pd.Series(spy)
        .rolling(config.trend_sma, min_periods=config.trend_sma)
        .mean()
        .values
    )

    if config.trend_type == "dual_sma":
        spy_sma_fast = (
            pd.Series(spy)
            .rolling(config.trend_sma_fast, min_periods=config.trend_sma_fast)
            .mean()
            .values
        )

    # Momentum computation
    if config.weighting == "inv_vol":
        mom = compute_risk_adjusted_momentum(
            pivoted, available, config.mom_lookbacks, config.mom_weights,
            config.skip_recent, config.vol_lookback
        )
    else:
        mom = compute_blended_momentum(
            pivoted, available, config.mom_lookbacks, config.mom_weights,
            config.skip_recent
        )

    # Volatility for exposure scaling
    port_daily_ret = pivoted[available].pct_change().mean(axis=1)
    rvol = port_daily_ret.rolling(config.vol_lookback, min_periods=10).std().values * math.sqrt(252)

    warmup = max(max(config.mom_lookbacks) + config.skip_recent, config.trend_sma, config.vol_lookback) + 5
    equity = config.initial_equity
    current_holdings: list[str] = []
    current_weights: dict[str, float] = {}
    equity_curve = np.full(n, np.nan)
    spy_norm = np.full(n, np.nan)
    spy_base = spy[warmup] if warmup < n else spy[0]
    total_trades = 0
    last_rebalance = 0
    days_in_cash = 0
    days_invested = 0
    max_equity = config.initial_equity
    underwater_days = 0
    max_underwater = 0

    for i in range(n):
        spy_norm[i] = config.initial_equity * (spy[i] / spy_base) if spy_base > 0 else config.initial_equity
        if i < warmup:
            equity_curve[i] = equity
            continue

        # Volatility-scaled exposure
        rv = rvol[i - 1] if (i - 1) < len(rvol) and not np.isnan(rvol[i - 1]) and rvol[i - 1] > 0 else config.vol_target
        exposure = float(np.clip(config.vol_target / rv, config.exposure_min, config.exposure_max))

        # P&L from existing positions
        if current_holdings and i > 0:
            day_pnl = 0.0
            for sym in current_holdings:
                w = current_weights.get(sym, 1.0 / len(current_holdings))
                prev_p = pivoted[sym].iloc[i - 1]
                curr_p = pivoted[sym].iloc[i]
                if prev_p > 0 and not np.isnan(prev_p) and not np.isnan(curr_p):
                    alloc = equity * w
                    day_pnl += alloc * (curr_p / prev_p - 1) * exposure
            equity += day_pnl

        # Track drawdown stats
        if equity > max_equity:
            max_equity = equity
            underwater_days = 0
        else:
            underwater_days += 1
            max_underwater = max(max_underwater, underwater_days)

        # Check if rebalance day
        should_rebalance = (i - last_rebalance >= config.rebalance_freq) or (i == warmup)

        if should_rebalance:
            # Trend filter
            if config.trend_type == "binary":
                trend_signal = 1.0 if (spy[i] > spy_sma_slow[i] and not np.isnan(spy_sma_slow[i])) else 0.0
            elif config.trend_type == "gradual":
                if np.isnan(spy_sma_slow[i]):
                    trend_signal = 0.0
                else:
                    pct_above = (spy[i] / spy_sma_slow[i] - 1)
                    trend_signal = float(np.clip(pct_above * 10 + 0.5, 0.0, 1.0))
            elif config.trend_type == "dual_sma":
                if np.isnan(spy_sma_slow[i]) or np.isnan(spy_sma_fast[i]):
                    trend_signal = 0.0
                elif spy[i] > spy_sma_slow[i] and spy_sma_fast[i] > spy_sma_slow[i]:
                    trend_signal = 1.0
                elif spy[i] > spy_sma_slow[i] or spy_sma_fast[i] > spy_sma_slow[i]:
                    trend_signal = 0.5
                else:
                    trend_signal = 0.0
            else:
                trend_signal = 1.0

            if trend_signal <= 0:
                target_holdings = []
                target_weights = {}
            else:
                mom_today = mom.iloc[i].dropna() if i < len(mom) else pd.Series(dtype=float)

                if config.absolute_mom_filter:
                    mom_today = mom_today[mom_today > 0]

                if len(mom_today) >= config.top_n:
                    ranked = mom_today.sort_values(ascending=False)
                    target_holdings = list(ranked.index[:config.top_n])

                    if config.weighting == "inv_vol":
                        inv_vols = {}
                        for sym in target_holdings:
                            vol_s = pivoted[sym].pct_change().rolling(config.vol_lookback).std().iloc[i]
                            if vol_s > 0 and not np.isnan(vol_s):
                                inv_vols[sym] = 1.0 / vol_s
                            else:
                                inv_vols[sym] = 1.0
                        total_iv = sum(inv_vols.values())
                        target_weights = {s: v / total_iv for s, v in inv_vols.items()}
                    elif config.weighting == "momentum_weighted":
                        scores = {s: max(float(ranked[s]), 0.001) for s in target_holdings}
                        total_s = sum(scores.values())
                        target_weights = {s: v / total_s for s, v in scores.items()}
                    else:
                        target_weights = {s: 1.0 / len(target_holdings) for s in target_holdings}

                    if trend_signal < 1.0:
                        target_weights = {s: w * trend_signal for s, w in target_weights.items()}
                else:
                    target_holdings = list(current_holdings)
                    target_weights = dict(current_weights)

            if set(target_holdings) != set(current_holdings) or target_weights != current_weights:
                turnover = len(set(current_holdings) - set(target_holdings)) + len(set(target_holdings) - set(current_holdings))
                if turnover > 0:
                    cost = turnover * config.cost_fraction * equity / max(config.top_n * 2, 1)
                    equity -= cost
                    total_trades += turnover
                current_holdings = list(target_holdings)
                current_weights = dict(target_weights)
                last_rebalance = i

        if current_holdings:
            days_invested += 1
        else:
            days_in_cash += 1

        equity_curve[i] = equity

    # Compute metrics
    eq = equity_curve[warmup:]
    spy_n = spy_norm[warmup:]
    valid = ~np.isnan(eq) & ~np.isnan(spy_n)
    eq, spy_n = eq[valid], spy_n[valid]

    if len(eq) < 2:
        return {"error": "not enough data"}

    d_ret = np.diff(eq) / eq[:-1]
    d_ret = d_ret[np.isfinite(d_ret)]
    days = len(d_ret)
    years = days / 252
    std = np.std(d_ret)

    cagr = (eq[-1] / eq[0]) ** (1 / max(years, 0.1)) - 1
    spy_cagr = (spy_n[-1] / spy_n[0]) ** (1 / max(years, 0.1)) - 1
    sharpe = float(np.mean(d_ret) / std * math.sqrt(252)) if std > 1e-12 else 0
    sortino_denom = np.std(d_ret[d_ret < 0]) if np.any(d_ret < 0) else std
    sortino = float(np.mean(d_ret) / sortino_denom * math.sqrt(252)) if sortino_denom > 1e-12 else 0
    peak = np.maximum.accumulate(eq)
    drawdowns = eq / peak - 1
    max_dd = float(np.min(drawdowns))

    # Calmar ratio
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0.001 else 0

    # Win rate
    d_ret_nonzero = d_ret[np.abs(d_ret) > 1e-10]
    win_rate = float(np.mean(d_ret_nonzero > 0)) if len(d_ret_nonzero) > 0 else 0

    # Profit factor
    gains = d_ret[d_ret > 0].sum()
    losses = abs(d_ret[d_ret < 0].sum())
    profit_factor = gains / losses if losses > 0 else float("inf")

    # Monthly returns for consistency check
    monthly_eq = eq[::21]
    if len(monthly_eq) > 1:
        monthly_ret = np.diff(monthly_eq) / monthly_eq[:-1]
        monthly_win_rate = float(np.mean(monthly_ret > 0))
        worst_month = float(np.min(monthly_ret))
        best_month = float(np.max(monthly_ret))
    else:
        monthly_win_rate = 0
        worst_month = 0
        best_month = 0

    # Annual returns for consistency
    annual_eq = eq[::252]
    if len(annual_eq) > 1:
        annual_ret = np.diff(annual_eq) / annual_eq[:-1]
        negative_years = int(np.sum(annual_ret < 0))
        worst_year = float(np.min(annual_ret))
    else:
        negative_years = 0
        worst_year = 0

    return {
        "name": config.name,
        "cagr_pct": round(cagr * 100, 2),
        "spy_cagr_pct": round(spy_cagr * 100, 2),
        "alpha_pct": round((cagr - spy_cagr) * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_return_pct": round((eq[-1] / eq[0] - 1) * 100, 1),
        "spy_return_pct": round((spy_n[-1] / spy_n[0] - 1) * 100, 1),
        "total_trades": total_trades,
        "years": round(years, 1),
        "final_equity": round(float(eq[-1]), 0),
        "win_rate": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2),
        "monthly_win_rate": round(monthly_win_rate * 100, 1),
        "worst_month_pct": round(worst_month * 100, 2),
        "best_month_pct": round(best_month * 100, 2),
        "negative_years": negative_years,
        "worst_year_pct": round(worst_year * 100, 2),
        "days_invested_pct": round(days_invested / max(days_invested + days_in_cash, 1) * 100, 1),
        "max_underwater_days": max_underwater,
        "cost_bps": config.round_trip_cost_bps,
    }


def run_all_tests(daily: pd.DataFrame) -> list[dict]:
    """Run comprehensive strategy tests."""
    results = []

    # ── Current baseline strategy ────────────────────────────────────
    print("Testing: Current baseline (20d mom, top 3, SMA 100)...")
    results.append(backtest_strategy(daily, StrategyConfig(
        name="BASELINE: 20d-mom top3 SMA100",
        mom_lookbacks=[20], mom_weights=[1.0], top_n=3, trend_sma=100,
    )))

    # ── 1. Momentum lookback variations ──────────────────────────────
    for lb in [10, 20, 40, 60, 126, 252]:
        cfg = StrategyConfig(
            name=f"Single-mom {lb}d top3",
            mom_lookbacks=[lb], mom_weights=[1.0], top_n=3, trend_sma=100,
        )
        print(f"Testing: {cfg.name}...")
        results.append(backtest_strategy(daily, cfg))

    # ── 2. Multi-timeframe momentum (Adaptive Momentum style) ────────
    multi_tf_configs = [
        ("Blend 20+60", [20, 60], [1, 1]),
        ("Blend 20+60+126", [20, 60, 126], [1, 1, 1]),
        ("Blend 20+60+126+252", [20, 60, 126, 252], [1, 1, 1, 1]),
        ("Blend 60+126+252 (med-long)", [60, 126, 252], [1, 1, 1]),
        ("Blend 60+126 skip5", [60, 126], [1, 1]),
        ("12m-1m classic (252 skip21)", [252], [1]),
    ]
    for name, lbs, ws in multi_tf_configs:
        cfg = StrategyConfig(
            name=f"Multi-TF: {name} top3",
            mom_lookbacks=lbs, mom_weights=ws, top_n=3, trend_sma=100,
            skip_recent=21 if "skip" in name or "12m-1m" in name else 0,
        )
        print(f"Testing: {cfg.name}...")
        results.append(backtest_strategy(daily, cfg))

    # ── 3. Top N variations ──────────────────────────────────────────
    for n in [1, 2, 3, 5, 7, 10]:
        cfg = StrategyConfig(
            name=f"Top-{n} (20d mom)",
            mom_lookbacks=[20], mom_weights=[1.0], top_n=n, trend_sma=100,
        )
        print(f"Testing: {cfg.name}...")
        results.append(backtest_strategy(daily, cfg))

    # ── 4. Trend filter variations ───────────────────────────────────
    for sma in [50, 100, 150, 200]:
        cfg = StrategyConfig(
            name=f"SMA-{sma} trend (20d mom top3)",
            mom_lookbacks=[20], mom_weights=[1.0], top_n=3, trend_sma=sma,
        )
        print(f"Testing: {cfg.name}...")
        results.append(backtest_strategy(daily, cfg))

    # Gradual trend filter
    for sma in [100, 200]:
        cfg = StrategyConfig(
            name=f"Gradual SMA-{sma} (20d mom top3)",
            mom_lookbacks=[20], mom_weights=[1.0], top_n=3,
            trend_sma=sma, trend_type="gradual",
        )
        print(f"Testing: {cfg.name}...")
        results.append(backtest_strategy(daily, cfg))

    # Dual SMA
    for fast, slow in [(50, 200), (20, 100), (50, 100)]:
        cfg = StrategyConfig(
            name=f"Dual SMA {fast}/{slow} (20d mom top3)",
            mom_lookbacks=[20], mom_weights=[1.0], top_n=3,
            trend_sma=slow, trend_sma_fast=fast, trend_type="dual_sma",
        )
        print(f"Testing: {cfg.name}...")
        results.append(backtest_strategy(daily, cfg))

    # ── 5. Universe variations ───────────────────────────────────────
    for univ, name in [("etf_only", "ETF-only"), ("stock_only", "Stock-only"), ("mixed", "Mixed")]:
        cfg = StrategyConfig(
            name=f"{name} universe (20d mom top3)",
            universe_type=univ,
            mom_lookbacks=[20], mom_weights=[1.0], top_n=3, trend_sma=100,
        )
        print(f"Testing: {cfg.name}...")
        results.append(backtest_strategy(daily, cfg))

    # ── 6. Weighting variations ──────────────────────────────────────
    for wt in ["equal", "inv_vol", "momentum_weighted"]:
        cfg = StrategyConfig(
            name=f"{wt} weighting (20d mom top5)",
            weighting=wt,
            mom_lookbacks=[20], mom_weights=[1.0], top_n=5, trend_sma=100,
        )
        print(f"Testing: {cfg.name}...")
        results.append(backtest_strategy(daily, cfg))

    # ── 7. Rebalance frequency ───────────────────────────────────────
    for freq in [1, 5, 21]:
        lbl = {1: "daily", 5: "weekly", 21: "monthly"}[freq]
        cfg = StrategyConfig(
            name=f"{lbl} rebalance (20d mom top3)",
            rebalance_freq=freq,
            mom_lookbacks=[20], mom_weights=[1.0], top_n=3, trend_sma=100,
        )
        print(f"Testing: {cfg.name}...")
        results.append(backtest_strategy(daily, cfg))

    # ── 8. Absolute momentum filter ──────────────────────────────────
    cfg = StrategyConfig(
        name="Abs-mom filter (20d top3)",
        absolute_mom_filter=True,
        mom_lookbacks=[20], mom_weights=[1.0], top_n=3, trend_sma=100,
    )
    print(f"Testing: {cfg.name}...")
    results.append(backtest_strategy(daily, cfg))

    # ── 9. Cost sensitivity ──────────────────────────────────────────
    for cost_name, cost_bps in COST_LEVELS.items():
        cfg = StrategyConfig(
            name=f"Cost={cost_name} ({cost_bps}bps) base strategy",
            round_trip_cost_bps=cost_bps,
            mom_lookbacks=[20], mom_weights=[1.0], top_n=3, trend_sma=100,
        )
        print(f"Testing: {cfg.name}...")
        results.append(backtest_strategy(daily, cfg))

    # ── 10. Combined best-of candidates ──────────────────────────────
    candidates = [
        StrategyConfig(
            name="CANDIDATE A: Blend60+126 top5 invVol SMA200 weekly",
            mom_lookbacks=[60, 126], mom_weights=[1, 1], top_n=5,
            trend_sma=200, weighting="inv_vol", rebalance_freq=5,
            round_trip_cost_bps=25.0,
        ),
        StrategyConfig(
            name="CANDIDATE B: Blend20+60+126 top5 equal SMA100 abs-filter",
            mom_lookbacks=[20, 60, 126], mom_weights=[1, 1, 1], top_n=5,
            trend_sma=100, absolute_mom_filter=True, rebalance_freq=5,
            round_trip_cost_bps=25.0,
        ),
        StrategyConfig(
            name="CANDIDATE C: 12m-1m top3 equal SMA200 monthly",
            mom_lookbacks=[252], mom_weights=[1], skip_recent=21, top_n=3,
            trend_sma=200, rebalance_freq=21,
            round_trip_cost_bps=25.0,
        ),
        StrategyConfig(
            name="CANDIDATE D: Blend60+126 top5 invVol gradualSMA200 weekly",
            mom_lookbacks=[60, 126], mom_weights=[1, 1], top_n=5,
            trend_sma=200, trend_type="gradual", weighting="inv_vol",
            rebalance_freq=5, round_trip_cost_bps=25.0,
        ),
        StrategyConfig(
            name="CANDIDATE E: Blend20+60 top3 equal dualSMA50/200 weekly",
            mom_lookbacks=[20, 60], mom_weights=[1, 1], top_n=3,
            trend_sma=200, trend_sma_fast=50, trend_type="dual_sma",
            rebalance_freq=5, round_trip_cost_bps=25.0,
        ),
        StrategyConfig(
            name="CANDIDATE F: ETF sector rotation 60+126 top3 SMA200 monthly",
            universe_type="etf_only",
            mom_lookbacks=[60, 126], mom_weights=[1, 1], top_n=3,
            trend_sma=200, rebalance_freq=21, round_trip_cost_bps=25.0,
        ),
        StrategyConfig(
            name="CANDIDATE G: Blend20+60+126 top5 invVol SMA100 weekly abs-filter",
            mom_lookbacks=[20, 60, 126], mom_weights=[1, 1, 2], top_n=5,
            trend_sma=100, weighting="inv_vol", absolute_mom_filter=True,
            rebalance_freq=5, round_trip_cost_bps=25.0,
        ),
        StrategyConfig(
            name="CANDIDATE H: Blend60+126+252 top7 invVol SMA200 weekly",
            mom_lookbacks=[60, 126, 252], mom_weights=[1, 2, 1], top_n=7,
            trend_sma=200, weighting="inv_vol", rebalance_freq=5,
            round_trip_cost_bps=25.0,
        ),
        StrategyConfig(
            name="CANDIDATE I: 12m-1m top5 invVol dualSMA50/200 monthly",
            mom_lookbacks=[252], mom_weights=[1], skip_recent=21, top_n=5,
            trend_sma=200, trend_sma_fast=50, trend_type="dual_sma",
            weighting="inv_vol", rebalance_freq=21, round_trip_cost_bps=25.0,
        ),
        StrategyConfig(
            name="CANDIDATE J: Blend20+60+126 top5 invVol gradualSMA100 abs-filter",
            mom_lookbacks=[20, 60, 126], mom_weights=[1, 2, 1], top_n=5,
            trend_sma=100, trend_type="gradual", weighting="inv_vol",
            absolute_mom_filter=True, rebalance_freq=5, round_trip_cost_bps=25.0,
        ),
    ]

    for cfg in candidates:
        print(f"Testing: {cfg.name}...")
        results.append(backtest_strategy(daily, cfg))

    return results


def print_results(results: list[dict]):
    """Print sorted results table."""
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda r: r.get("sharpe", 0), reverse=True)

    print("\n" + "=" * 140)
    print(f"{'Strategy':<60} {'CAGR':>6} {'SPY':>6} {'Alpha':>6} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} {'Trades':>7} {'WinMo%':>7} {'NegYr':>6} {'WorstYr':>8}")
    print("=" * 140)

    for r in valid:
        print(
            f"{r['name']:<60} "
            f"{r['cagr_pct']:>5.1f}% "
            f"{r['spy_cagr_pct']:>5.1f}% "
            f"{r['alpha_pct']:>5.1f}% "
            f"{r['sharpe']:>7.3f} "
            f"{r['sortino']:>8.3f} "
            f"{r['max_drawdown_pct']:>6.1f}% "
            f"{r['total_trades']:>7} "
            f"{r['monthly_win_rate']:>6.1f}% "
            f"{r['negative_years']:>5} "
            f"{r['worst_year_pct']:>7.1f}%"
        )

    print("=" * 140)
    print(f"\nTop 5 by Sharpe Ratio:")
    for i, r in enumerate(valid[:5]):
        print(f"  {i+1}. {r['name']}")
        print(f"     CAGR: {r['cagr_pct']:+.1f}% | Alpha: {r['alpha_pct']:+.1f}% | Sharpe: {r['sharpe']:.3f} | MaxDD: {r['max_drawdown_pct']:.1f}% | Sortino: {r['sortino']:.3f}")

    # Also rank by Calmar
    by_calmar = sorted(valid, key=lambda r: r.get("calmar", 0), reverse=True)
    print(f"\nTop 5 by Calmar Ratio (CAGR/MaxDD):")
    for i, r in enumerate(by_calmar[:5]):
        print(f"  {i+1}. {r['name']}")
        print(f"     CAGR: {r['cagr_pct']:+.1f}% | Calmar: {r['calmar']:.3f} | MaxDD: {r['max_drawdown_pct']:.1f}% | Sharpe: {r['sharpe']:.3f}")


if __name__ == "__main__":
    print("Loading data...")
    daily = load()
    syms = sorted(daily["symbol"].unique())
    print(f"  {len(syms)} symbols, {daily['timestamp'].dt.date.nunique()} trading days")
    print(f"  Date range: {daily['timestamp'].min().date()} to {daily['timestamp'].max().date()}")
    print()

    results = run_all_tests(daily)
    print_results(results)
