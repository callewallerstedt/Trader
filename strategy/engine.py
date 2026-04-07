"""
Momentum rotation strategy engine.

Strategy: Buy top-N stocks by 20-day momentum when SPY > 100-day SMA.
          Go to cash when SPY < 100-day SMA. Volatility-scale exposure.

Execution: Run at 3:45 PM ET, submit MOC orders, fill at 4 PM close.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

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
    momentum_lookback: int = 20
    trend_sma_period: int = 100
    top_n: int = 3
    vol_lookback: int = 20
    vol_target: float = 0.15
    exposure_min: float = 0.5
    exposure_max: float = 1.3
    round_trip_cost_bps: float = 20.0
    initial_equity: float = 100000.0

    @property
    def cost_fraction(self) -> float:
        return self.round_trip_cost_bps / 10000


def compute_signal(daily: pd.DataFrame, config: Config | None = None) -> dict:
    """
    Compute today's target portfolio from price history.

    Returns a dict with:
      action, target_holdings, momentum_scores, spy_price, spy_sma, trend,
      date, exposure, spy_history (last 150 days of SPY + SMA for charting)
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
    trend_up = latest_spy > latest_sma

    spy_hist = _build_spy_history(spy, spy_sma, n=150)

    spy_pct_vs_sma = (latest_spy / latest_sma - 1) * 100 if latest_sma > 0 else 0

    port_daily_ret = pivoted[tradeable].pct_change().mean(axis=1)
    rvol_series = port_daily_ret.rolling(config.vol_lookback, min_periods=10).std() * math.sqrt(252)
    rv = float(rvol_series.iloc[-1]) if not np.isnan(rvol_series.iloc[-1]) else config.vol_target
    exposure = float(np.clip(config.vol_target / rv, config.exposure_min, config.exposure_max)) if rv > 0 else 1.0

    result = {
        "spy_price": latest_spy,
        "spy_sma": latest_sma,
        "spy_pct_vs_sma": round(spy_pct_vs_sma, 2),
        "trend": "UP" if trend_up else "DOWN",
        "date": str(pivoted.index[-1].date()),
        "exposure": round(exposure, 3),
        "realized_vol": round(rv * 100, 1),
        "spy_history": spy_hist,
    }

    if not trend_up:
        days_below = _days_since_cross(spy, spy_sma, direction="below")
        result["action"] = "go_to_cash"
        result["target_holdings"] = []
        result["reason"] = f"SPY ${latest_spy:.2f} below SMA ${latest_sma:.2f}"
        result["days_in_regime"] = days_below
        return result

    days_above = _days_since_cross(spy, spy_sma, direction="above")
    mom = pivoted[tradeable].pct_change(config.momentum_lookback)
    latest_mom = mom.iloc[-1].dropna()

    if len(latest_mom) < config.top_n:
        result["action"] = "wait"
        result["reason"] = "not enough momentum data"
        return result

    ranked = latest_mom.sort_values(ascending=False)
    target = list(ranked.index[: config.top_n])

    result["action"] = "hold"
    result["target_holdings"] = target
    result["days_in_regime"] = days_above
    result["momentum_scores"] = {
        sym: round(float(ranked[sym]) * 100, 2) for sym in ranked.index[:10]
    }
    result["prices"] = {sym: float(pivoted[sym].iloc[-1]) for sym in target}
    return result


def _build_spy_history(spy: pd.Series, spy_sma: pd.Series, n: int = 150) -> list[dict]:
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
    if direction == "below":
        condition = ~above
    else:
        condition = above

    count = 0
    for val in reversed(condition.values):
        if val and not np.isnan(val):
            count += 1
        else:
            break
    return count


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
    mom = pivoted[tradeable].pct_change(config.momentum_lookback)
    port_daily_ret = pivoted[tradeable].pct_change().mean(axis=1)
    rvol = port_daily_ret.rolling(config.vol_lookback, min_periods=10).std().values * math.sqrt(252)

    warmup = max(config.momentum_lookback, config.trend_sma_period, config.vol_lookback) + 5
    equity = config.initial_equity
    current_holdings: list[str] = []
    equity_curve = np.full(n, np.nan)
    spy_norm = np.full(n, np.nan)
    spy_base = spy[warmup] if warmup < n else spy[0]
    total_trades = 0

    for i in range(n):
        spy_norm[i] = config.initial_equity * (spy[i] / spy_base) if spy_base > 0 else config.initial_equity
        if i < warmup:
            equity_curve[i] = equity
            continue

        rv = rvol[i - 1] if (i - 1) < len(rvol) and not np.isnan(rvol[i - 1]) and rvol[i - 1] > 0 else config.vol_target
        exposure = float(np.clip(config.vol_target / rv, config.exposure_min, config.exposure_max))

        if current_holdings and i > 0:
            day_pnl = 0.0
            for sym in current_holdings:
                prev_p = pivoted[sym].iloc[i - 1]
                curr_p = pivoted[sym].iloc[i]
                if prev_p > 0 and not np.isnan(prev_p) and not np.isnan(curr_p):
                    alloc = equity / len(current_holdings)
                    day_pnl += alloc * (curr_p / prev_p - 1) * exposure
            equity += day_pnl

        sig_trend_up = spy[i] > spy_sma[i] if not np.isnan(spy_sma[i]) else False
        if not sig_trend_up:
            target_holdings = []
        else:
            mom_today = mom.iloc[i].dropna() if i < len(mom) else pd.Series(dtype=float)
            if len(mom_today) >= config.top_n:
                ranked = mom_today.sort_values(ascending=False)
                target_holdings = list(ranked.index[: config.top_n])
            else:
                target_holdings = list(current_holdings)

        if set(target_holdings) != set(current_holdings):
            turnover = len(set(current_holdings) - set(target_holdings)) + len(set(target_holdings) - set(current_holdings))
            if turnover > 0:
                cost = turnover * config.cost_fraction * equity / max(config.top_n * 2, 1)
                equity -= cost
                total_trades += turnover
            current_holdings = list(target_holdings)

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

    return {
        "cagr_pct": round(cagr * 100, 1),
        "spy_cagr_pct": round(spy_cagr * 100, 1),
        "alpha_pct": round((cagr - spy_cagr) * 100, 1),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "total_return_pct": round((eq[-1] / eq[0] - 1) * 100, 1),
        "spy_return_pct": round((spy_n[-1] / spy_n[0] - 1) * 100, 1),
        "total_trades": total_trades,
        "years": round(years, 1),
        "final_equity": round(float(eq[-1]), 0),
    }
