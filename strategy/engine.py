"""
Multi-Timeframe Momentum Rotation with Gradual Trend Filter.

Strategy: Blend 20/60/126-day momentum to rank stocks. Hold top-5 with
          positive absolute momentum. Gradual SMA-200 trend filter scales
          equity exposure between 0-100% based on SPY distance from SMA.
          Volatility-scaled position sizing targets 15% annualized vol.

Execution: Run at 3:45 PM ET, submit MOC orders, fill at 4 PM close.
           Rebalance weekly (every 5 trading days).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


UNIVERSE = [
    "SPY",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY", "XLC",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    "JPM", "V", "UNH", "JNJ", "PG", "HD", "MA",
    "TSLA", "AMD", "NFLX",
    "XOM", "INTC", "BA", "PFE", "GE", "T",
    "EFA", "EEM", "GLD", "TLT",
]

NON_TRADEABLE = {"SPY", "DIA", "IWM", "TLT", "GLD", "EFA", "EEM"}


@dataclass
class Config:
    mom_lookbacks: list[int] = field(default_factory=lambda: [20, 60, 126])
    mom_weights: list[float] = field(default_factory=lambda: [1.0, 2.0, 1.0])
    top_n: int = 5
    trend_sma_period: int = 200
    trend_type: str = "gradual"
    absolute_mom_filter: bool = True
    rebalance_freq: int = 5
    vol_lookback: int = 20
    vol_target: float = 0.15
    exposure_min: float = 0.5
    exposure_max: float = 1.3
    round_trip_cost_bps: float = 25.0
    initial_equity: float = 100000.0

    @property
    def cost_fraction(self) -> float:
        return self.round_trip_cost_bps / 10000


def _blended_momentum(
    pivoted: pd.DataFrame,
    tradeable: list[str],
    lookbacks: list[int],
    weights: list[float],
) -> pd.DataFrame:
    """Weighted blend of multiple momentum lookback periods."""
    frames = []
    for lb, w in zip(lookbacks, weights):
        frames.append(pivoted[tradeable].pct_change(lb) * w)
    return sum(frames) / sum(weights)


def _trend_signal_gradual(spy_price: float, spy_sma: float) -> float:
    """Gradual trend filter: 0.0 (deep downtrend) to 1.0 (strong uptrend).

    Linear ramp from -5% below SMA (trend=0) to +5% above SMA (trend=1),
    centered at SMA crossing (trend=0.5).
    """
    if spy_sma <= 0:
        return 0.0
    pct = spy_price / spy_sma - 1.0
    return float(np.clip(pct * 10.0 + 0.5, 0.0, 1.0))


def _build_spy_history(spy: pd.Series, spy_sma: pd.Series, n: int = 200) -> list[dict]:
    """Build last n days of SPY price + SMA for charting."""
    tail_spy = spy.iloc[-n:]
    tail_sma = spy_sma.iloc[-n:]
    history = []
    for dt, price in tail_spy.items():
        sma_val = tail_sma.get(dt)
        if np.isnan(price):
            continue
        history.append({
            "date": str(dt.date()),
            "price": round(float(price), 2),
            "sma": round(float(sma_val), 2) if sma_val is not None and not np.isnan(sma_val) else None,
        })
    return history


def _days_since_cross(spy: pd.Series, spy_sma: pd.Series, direction: str) -> int:
    """Count trading days since SPY crossed above/below SMA."""
    above = spy > spy_sma
    condition = ~above if direction == "below" else above
    count = 0
    for val in reversed(condition.values):
        if val and not np.isnan(val):
            count += 1
        else:
            break
    return count


def compute_signal(daily: pd.DataFrame, config: Config | None = None) -> dict:
    """
    Compute today's target portfolio from price history.

    Returns a dict with: action, target_holdings, momentum_scores, spy_price,
    spy_sma, trend, trend_strength, date, exposure, spy_history, ...
    """
    config = config or Config()
    tradeable = [s for s in daily["symbol"].unique() if s not in NON_TRADEABLE]

    pivoted = (
        daily.pivot_table(index="timestamp", columns="symbol", values="close")
        .sort_index()
        .ffill()
    )

    if "SPY" not in pivoted.columns:
        return {"action": "error", "reason": "SPY not in data"}

    spy = pivoted["SPY"]
    spy_sma = spy.rolling(config.trend_sma_period, min_periods=config.trend_sma_period).mean()

    if spy_sma.iloc[-1] is None or np.isnan(spy_sma.iloc[-1]):
        return {"action": "wait", "reason": "not enough history for SMA"}

    latest_spy = float(spy.iloc[-1])
    latest_sma = float(spy_sma.iloc[-1])
    trend_strength = _trend_signal_gradual(latest_spy, latest_sma)
    trend_up = trend_strength > 0

    spy_hist = _build_spy_history(spy, spy_sma, n=200)
    spy_pct_vs_sma = (latest_spy / latest_sma - 1) * 100 if latest_sma > 0 else 0

    port_daily_ret = pivoted[tradeable].pct_change().mean(axis=1)
    rvol_series = port_daily_ret.rolling(config.vol_lookback, min_periods=10).std() * math.sqrt(252)
    rv = float(rvol_series.iloc[-1]) if not np.isnan(rvol_series.iloc[-1]) else config.vol_target
    exposure = float(np.clip(config.vol_target / rv, config.exposure_min, config.exposure_max)) if rv > 0 else 1.0

    mom = _blended_momentum(pivoted, tradeable, config.mom_lookbacks, config.mom_weights)
    lb_str = "+".join(str(lb) for lb in config.mom_lookbacks)

    result = {
        "spy_price": latest_spy,
        "spy_sma": latest_sma,
        "spy_pct_vs_sma": round(spy_pct_vs_sma, 2),
        "trend": "UP" if latest_spy > latest_sma else "DOWN",
        "trend_strength": round(trend_strength * 100, 1),
        "date": str(pivoted.index[-1].date()),
        "exposure": round(exposure, 3),
        "realized_vol": round(rv * 100, 1),
        "spy_history": spy_hist,
        "strategy": f"Multi-TF Momentum ({lb_str}d) top-{config.top_n}",
    }

    all_mom = mom.iloc[-1].dropna().sort_values(ascending=False)
    all_prices = {sym: float(pivoted[sym].iloc[-1]) for sym in tradeable if sym in pivoted.columns}

    result["all_momentum"] = {
        sym: round(float(all_mom[sym]) * 100, 2) for sym in all_mom.index
    }
    result["all_prices"] = all_prices

    if not trend_up:
        days_below = _days_since_cross(spy, spy_sma, direction="below")
        result["action"] = "go_to_cash"
        result["target_holdings"] = []
        result["reason"] = f"SPY ${latest_spy:.2f} below SMA ${latest_sma:.2f} (trend strength: {trend_strength:.0%})"
        result["days_in_regime"] = days_below
        result["effective_exposure"] = 0.0
        return result

    days_above = _days_since_cross(spy, spy_sma, direction="above")
    latest_mom = all_mom.copy()

    if config.absolute_mom_filter:
        latest_mom = latest_mom[latest_mom > 0]

    if len(latest_mom) < config.top_n:
        result["action"] = "wait"
        result["reason"] = f"only {len(latest_mom)} stocks with positive momentum"
        return result

    ranked = latest_mom.sort_values(ascending=False)
    target = list(ranked.index[:config.top_n])

    effective_exposure = exposure * trend_strength if trend_strength < 1.0 else exposure
    result["action"] = "hold"
    result["target_holdings"] = target
    result["days_in_regime"] = days_above
    result["trend_exposure"] = round(trend_strength, 3)
    result["effective_exposure"] = round(effective_exposure, 3)
    result["momentum_scores"] = {
        sym: round(float(ranked[sym]) * 100, 2) for sym in ranked.index[:10]
    }
    result["prices"] = {sym: float(pivoted[sym].iloc[-1]) for sym in target}
    return result


def backtest(daily: pd.DataFrame, config: Config | None = None) -> dict:
    """Run full backtest. Returns metrics dict."""
    config = config or Config()
    symbols = sorted(daily["symbol"].unique())
    tradeable = [s for s in symbols if s not in NON_TRADEABLE]

    pivoted = (
        daily.pivot_table(index="timestamp", columns="symbol", values="close")
        .sort_index()
        .ffill()
    )
    dates = pivoted.index.tolist()
    n = len(dates)

    if "SPY" not in pivoted.columns:
        raise ValueError("SPY required")

    spy = pivoted["SPY"].values
    spy_sma = (
        pd.Series(spy)
        .rolling(config.trend_sma_period, min_periods=config.trend_sma_period)
        .mean()
        .values
    )

    mom = _blended_momentum(pivoted, tradeable, config.mom_lookbacks, config.mom_weights)
    port_daily_ret = pivoted[tradeable].pct_change().mean(axis=1)
    rvol = port_daily_ret.rolling(config.vol_lookback, min_periods=10).std().values * math.sqrt(252)

    warmup = max(max(config.mom_lookbacks), config.trend_sma_period, config.vol_lookback) + 5
    equity = config.initial_equity
    current_holdings: list[str] = []
    equity_curve = np.full(n, np.nan)
    spy_norm = np.full(n, np.nan)
    spy_base = spy[warmup] if warmup < n else spy[0]
    total_trades = 0
    last_rebalance = 0

    current_trend_strength = 1.0

    for i in range(n):
        spy_norm[i] = config.initial_equity * (spy[i] / spy_base) if spy_base > 0 else config.initial_equity
        if i < warmup:
            equity_curve[i] = equity
            continue

        rv = rvol[i - 1] if (i - 1) < len(rvol) and not np.isnan(rvol[i - 1]) and rvol[i - 1] > 0 else config.vol_target
        exposure = float(np.clip(config.vol_target / rv, config.exposure_min, config.exposure_max))

        # Apply gradual trend factor to daily exposure
        effective_exposure = exposure * current_trend_strength

        if current_holdings and i > 0:
            day_pnl = 0.0
            for sym in current_holdings:
                prev_p = pivoted[sym].iloc[i - 1]
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
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.min(eq / peak - 1))

    monthly_eq = eq[::21]
    if len(monthly_eq) > 1:
        monthly_ret = np.diff(monthly_eq) / monthly_eq[:-1]
        monthly_win_rate = float(np.mean(monthly_ret > 0))
    else:
        monthly_win_rate = 0

    return {
        "cagr_pct": round(cagr * 100, 1),
        "spy_cagr_pct": round(spy_cagr * 100, 1),
        "alpha_pct": round((cagr - spy_cagr) * 100, 1),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "total_return_pct": round((eq[-1] / eq[0] - 1) * 100, 1),
        "spy_return_pct": round((spy_n[-1] / spy_n[0] - 1) * 100, 1),
        "monthly_win_rate_pct": round(monthly_win_rate * 100, 1),
        "total_trades": total_trades,
        "years": round(years, 1),
        "final_equity": round(float(eq[-1]), 0),
    }
