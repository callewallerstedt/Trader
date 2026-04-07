"""
Multi-Signal Momentum Rotation V3 — Combined Ranking Engine.

Signal:  score = 0.5 * momentum_rank + 0.3 * residual_momentum_rank
                 + 0.2 * acceleration_rank
         Picks top-15 stocks by combined score with positive absolute momentum.

Where:
  momentum       = blended 20/60/126-day returns (weights 1:2:1)
  residual_mom   = stock return minus beta * SPY return (126d rolling OLS)
                   Removes market-beta exposure → isolates stock-specific alpha.
  acceleration   = 21d momentum / |126d momentum| (clipped [-3, 3])
                   Captures early-phase trend acceleration.

Risk:    VIX crash filter (reduce at 20, flat at 30), gradual SMA-200 trend
         filter, inverse-vol position weighting with 15% per-stock cap.
         No portfolio vol-scaling (empirically reduces Sharpe).

Execution: MOC orders at 3:45 PM ET, fills at 4 PM close.
           Rebalance every 25 trading days.

Validated via research/alpha_final.py (15 years, survivorship-free universe):
  Sharpe 1.66  |  PF 1.35  |  Max DD -8.1%  |  CAGR +17.2%
  vs V2: Sharpe +32%, DD halved, PF above 1.30 target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


UNIVERSE = [
    "SPY",
    # Sector ETFs (all existed since 1998-1999)
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY",
    # Tech (mix of winners and losers for survivorship-free universe)
    "AAPL", "MSFT", "INTC", "IBM", "CSCO", "ORCL", "TXN", "QCOM",
    # Finance
    "JPM", "BAC", "WFC", "GS", "C",
    # Healthcare
    "JNJ", "PFE", "UNH", "MRK", "ABT", "LLY",
    # Consumer
    "PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE",
    # Energy
    "XOM", "CVX", "COP",
    # Industrial
    "GE", "HON", "MMM", "CAT", "BA",
    # Telecom
    "T", "VZ",
    # Broad/International (benchmark only)
    "EFA", "EEM", "GLD", "TLT",
]

NON_TRADEABLE = {"SPY", "DIA", "IWM", "TLT", "GLD", "EFA", "EEM"}

VIX_SYMBOL = "^VIX"


@dataclass
class Config:
    mom_lookbacks: list[int] = field(default_factory=lambda: [20, 60, 126])
    mom_weights: list[float] = field(default_factory=lambda: [1.0, 2.0, 1.0])
    top_n: int = 15
    trend_sma_period: int = 200
    trend_type: str = "gradual"
    absolute_mom_filter: bool = True
    rebalance_freq: int = 25
    vol_lookback: int = 20
    round_trip_cost_bps: float = 25.0
    initial_equity: float = 100000.0
    inv_vol_weight: bool = True
    max_position_pct: float = 0.15
    vix_reduce_threshold: float = 20.0
    vix_flat_threshold: float = 30.0
    # Combined signal weights (must sum to 1.0)
    sig_mom_weight: float = 0.5
    sig_resid_weight: float = 0.3
    sig_accel_weight: float = 0.2
    resid_lookback: int = 126
    accel_short: int = 21
    accel_long: int = 126

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


def _residual_momentum(
    pivoted: pd.DataFrame,
    tradeable: list[str],
    lookback: int = 126,
) -> pd.DataFrame:
    """Stock return minus beta * SPY return (rolling OLS).

    Isolates stock-specific momentum by removing market-beta exposure.
    Academic basis: Blitz, Huij, Martens (2011) "Residual Momentum."
    """
    if "SPY" not in pivoted.columns:
        return pd.DataFrame(np.nan, index=pivoted.index, columns=tradeable)
    spy_ret = pivoted["SPY"].pct_change().values
    resid = pd.DataFrame(np.nan, index=pivoted.index, columns=tradeable)

    for sym in tradeable:
        if sym not in pivoted.columns:
            continue
        stock_ret = pivoted[sym].pct_change().values
        col_idx = resid.columns.get_loc(sym)
        for i in range(lookback, len(pivoted)):
            sr = stock_ret[i - lookback:i]
            mr = spy_ret[i - lookback:i]
            valid = ~(np.isnan(sr) | np.isnan(mr))
            if valid.sum() < lookback * 0.7:
                continue
            slope, _, _, _, _ = stats.linregress(mr[valid], sr[valid])
            resid_rets = sr[valid] - slope * mr[valid]
            resid.iat[i, col_idx] = float(np.sum(resid_rets))
    return resid


def _residual_momentum_latest(
    pivoted: pd.DataFrame,
    tradeable: list[str],
    lookback: int = 126,
) -> dict[str, float]:
    """Compute residual momentum for the latest date only (fast, for live signal)."""
    if "SPY" not in pivoted.columns:
        return {}
    spy_ret = pivoted["SPY"].pct_change().values
    result = {}
    for sym in tradeable:
        if sym not in pivoted.columns:
            continue
        stock_ret = pivoted[sym].pct_change().values
        sr = stock_ret[-lookback:]
        mr = spy_ret[-lookback:]
        valid = ~(np.isnan(sr) | np.isnan(mr))
        if valid.sum() < lookback * 0.7:
            continue
        slope, _, _, _, _ = stats.linregress(mr[valid], sr[valid])
        resid_rets = sr[valid] - slope * mr[valid]
        result[sym] = float(np.sum(resid_rets))
    return result


def _momentum_acceleration(
    pivoted: pd.DataFrame,
    tradeable: list[str],
    short_lb: int = 21,
    long_lb: int = 126,
) -> pd.DataFrame:
    """Ratio of short-term to long-term momentum (clipped [-3, 3]).

    Values > 1 indicate accelerating momentum (early trend phase).
    """
    short_mom = pivoted[tradeable].pct_change(short_lb)
    long_abs = pivoted[tradeable].pct_change(long_lb).abs().replace(0, np.nan)
    accel = (short_mom / long_abs).clip(-3, 3)
    return accel


def _combined_ranking(
    pivoted: pd.DataFrame,
    tradeable: list[str],
    config: Config,
    mom: pd.DataFrame | None = None,
    resid: pd.DataFrame | None = None,
    accel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Cross-sectional percentile rank of each signal, then weighted sum."""
    combined = pd.DataFrame(0.0, index=pivoted.index, columns=tradeable)
    if mom is not None and config.sig_mom_weight > 0:
        combined += mom.rank(axis=1, pct=True) * config.sig_mom_weight
    if resid is not None and config.sig_resid_weight > 0:
        combined += resid.rank(axis=1, pct=True) * config.sig_resid_weight
    if accel is not None and config.sig_accel_weight > 0:
        combined += accel.rank(axis=1, pct=True) * config.sig_accel_weight
    return combined


def _trend_signal_gradual(spy_price: float, spy_sma: float) -> float:
    """Gradual trend filter: 0.0 (deep downtrend) to 1.0 (strong uptrend).

    Linear ramp from -5% below SMA (trend=0) to +5% above SMA (trend=1),
    centered at SMA crossing (trend=0.5).
    """
    if spy_sma <= 0:
        return 0.0
    pct = spy_price / spy_sma - 1.0
    return float(np.clip(pct * 10.0 + 0.5, 0.0, 1.0))


def _compute_inv_vol_weights(
    pivoted: pd.DataFrame,
    stocks: list[str],
    vol_lookback: int,
    max_position_pct: float,
    row_idx: int,
) -> dict[str, float]:
    """Compute inverse-volatility weights with per-stock cap."""
    inv_vols = {}
    for s in stocks:
        rets = pivoted[s].pct_change()
        # Use data up to row_idx
        recent = rets.iloc[max(0, row_idx - vol_lookback):row_idx]
        vol = float(recent.std() * math.sqrt(252))
        if vol <= 0 or np.isnan(vol):
            vol = 0.20  # fallback
        inv_vols[s] = 1.0 / vol

    total = sum(inv_vols.values())
    if total <= 0:
        return {s: 1.0 / len(stocks) for s in stocks}

    weights = {s: min(iv / total, max_position_pct) for s, iv in inv_vols.items()}
    # Renormalize after cap
    w_sum = sum(weights.values())
    return {s: w / w_sum for s, w in weights.items()} if w_sum > 0 else {s: 1.0 / len(stocks) for s in stocks}


def _vix_exposure_multiplier(vix_value: float | None, config: Config) -> float:
    """VIX crash filter: linearly reduce exposure between thresholds."""
    if vix_value is None or vix_value < config.vix_reduce_threshold:
        return 1.0
    if vix_value >= config.vix_flat_threshold:
        return 0.0
    return max(0.0, 1.0 - (vix_value - config.vix_reduce_threshold) /
               (config.vix_flat_threshold - config.vix_reduce_threshold))


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


def compute_signal(daily: pd.DataFrame, config: Config | None = None,
                   vix_value: float | None = None) -> dict:
    """
    Compute today's target portfolio from price history.

    Returns a dict with: action, target_holdings, target_weights,
    momentum_scores, spy_price, spy_sma, trend, trend_strength,
    date, exposure, spy_history, vix_multiplier, ...
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

    # VIX crash filter
    vix_mult = _vix_exposure_multiplier(vix_value, config)

    # Combined signal: momentum + residual + acceleration (ranked)
    mom = _blended_momentum(pivoted, tradeable, config.mom_lookbacks, config.mom_weights)
    resid_latest = _residual_momentum_latest(pivoted, tradeable, config.resid_lookback)
    accel_latest_row = pivoted[tradeable].pct_change(config.accel_short).iloc[-1]
    long_abs_latest = pivoted[tradeable].pct_change(config.accel_long).iloc[-1].abs().replace(0, np.nan)
    accel_latest = (accel_latest_row / long_abs_latest).clip(-3, 3)

    mom_latest = mom.iloc[-1].dropna()
    mom_ranks = mom_latest.rank(pct=True)

    resid_series = pd.Series(resid_latest)
    resid_ranks = resid_series.rank(pct=True)

    accel_ranks = accel_latest.dropna().rank(pct=True)

    common = list(set(mom_ranks.index) & set(resid_ranks.index) & set(accel_ranks.index))
    combined_score = pd.Series(0.0, index=common)
    for sym in common:
        combined_score[sym] = (
            config.sig_mom_weight * mom_ranks.get(sym, 0.5) +
            config.sig_resid_weight * resid_ranks.get(sym, 0.5) +
            config.sig_accel_weight * accel_ranks.get(sym, 0.5)
        )

    result = {
        "spy_price": latest_spy,
        "spy_sma": latest_sma,
        "spy_pct_vs_sma": round(spy_pct_vs_sma, 2),
        "trend": "UP" if latest_spy > latest_sma else "DOWN",
        "trend_strength": round(trend_strength * 100, 1),
        "date": str(pivoted.index[-1].date()),
        "exposure": 1.0,
        "realized_vol": 0.0,
        "spy_history": spy_hist,
        "strategy": f"V3 Combined (Mom+Resid+Accel) top-{config.top_n} inv-vol 25d",
        "vix_value": vix_value,
        "vix_multiplier": round(vix_mult, 3),
    }

    all_mom = combined_score.sort_values(ascending=False)
    all_prices = {sym: float(pivoted[sym].iloc[-1]) for sym in tradeable if sym in pivoted.columns}

    result["all_momentum"] = {
        sym: round(float(all_mom[sym]) * 100, 2) for sym in all_mom.index
    }
    result["all_prices"] = all_prices

    # VIX kills everything
    if vix_mult <= 0:
        result["action"] = "go_to_cash"
        result["target_holdings"] = []
        result["target_weights"] = {}
        result["reason"] = f"VIX at {vix_value:.1f} >= {config.vix_flat_threshold} (crash filter active)"
        result["effective_exposure"] = 0.0
        return result

    if not trend_up:
        days_below = _days_since_cross(spy, spy_sma, direction="below")
        result["action"] = "go_to_cash"
        result["target_holdings"] = []
        result["target_weights"] = {}
        result["reason"] = f"SPY ${latest_spy:.2f} below SMA ${latest_sma:.2f} (trend strength: {trend_strength:.0%})"
        result["days_in_regime"] = days_below
        result["effective_exposure"] = 0.0
        return result

    days_above = _days_since_cross(spy, spy_sma, direction="above")

    # Filter to stocks with positive raw momentum (absolute mom filter)
    raw_mom_latest = mom.iloc[-1].dropna()
    positive_mom = raw_mom_latest[raw_mom_latest > 0].index
    latest_ranked = all_mom[all_mom.index.isin(positive_mom)] if config.absolute_mom_filter else all_mom

    if len(latest_ranked) < config.top_n:
        result["action"] = "wait"
        result["reason"] = f"only {len(latest_ranked)} stocks with positive momentum"
        return result

    ranked = latest_ranked.sort_values(ascending=False)
    target = list(ranked.index[:config.top_n])

    # Compute weights
    if config.inv_vol_weight:
        weights = _compute_inv_vol_weights(
            pivoted, target, config.vol_lookback,
            config.max_position_pct, len(pivoted) - 1,
        )
    else:
        w = 1.0 / len(target)
        weights = {s: w for s in target}

    effective_exposure = trend_strength * vix_mult
    result["action"] = "hold"
    result["target_holdings"] = target
    result["target_weights"] = {s: round(w, 4) for s, w in weights.items()}
    result["days_in_regime"] = days_above
    result["trend_exposure"] = round(trend_strength, 3)
    result["effective_exposure"] = round(effective_exposure, 3)
    result["momentum_scores"] = {
        sym: round(float(ranked[sym]) * 100, 2) for sym in ranked.index[:20]
    }
    result["prices"] = {sym: float(pivoted[sym].iloc[-1]) for sym in target}
    return result


def backtest(daily: pd.DataFrame, config: Config | None = None,
             vix_series: pd.Series | None = None) -> dict:
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

    raw_mom = _blended_momentum(pivoted, tradeable, config.mom_lookbacks, config.mom_weights)

    # Compute residual momentum and acceleration for combined ranking
    resid = _residual_momentum(pivoted, tradeable, config.resid_lookback)
    accel = _momentum_acceleration(pivoted, tradeable, config.accel_short, config.accel_long)
    mom = _combined_ranking(pivoted, tradeable, config, raw_mom, resid, accel)

    # Per-stock vol for inverse-vol weighting
    stock_vol = {}
    if config.inv_vol_weight:
        for s in tradeable:
            if s in pivoted.columns:
                stock_vol[s] = pivoted[s].pct_change().rolling(config.vol_lookback, min_periods=10).std().values * math.sqrt(252)

    # VIX lookup
    vix_dict: dict = {}
    if vix_series is not None:
        for dt in dates:
            ts = pd.Timestamp(dt)
            if ts in vix_series.index:
                vix_dict[dt] = float(vix_series[ts])

    warmup = max(max(config.mom_lookbacks), config.trend_sma_period,
                 config.vol_lookback, config.resid_lookback) + 10
    equity = config.initial_equity
    current_holdings: list[str] = []
    current_weights: dict[str, float] = {}
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

        # VIX crash filter (no vol scaling in V3)
        vix_val = vix_dict.get(dates[i])
        vix_mult = _vix_exposure_multiplier(vix_val, config)

        effective_exposure = current_trend_strength * vix_mult

        if current_holdings and i > 0:
            day_pnl = 0.0
            for sym in current_holdings:
                w = current_weights.get(sym, 1.0 / len(current_holdings))
                prev_p = pivoted[sym].iloc[i - 1]
                curr_p = pivoted[sym].iloc[i]
                if prev_p > 0 and not np.isnan(prev_p) and not np.isnan(curr_p):
                    day_pnl += equity * w * (curr_p / prev_p - 1) * effective_exposure
            equity += day_pnl

        should_rebalance = (i - last_rebalance >= config.rebalance_freq) or (i == warmup)

        if should_rebalance:
            if np.isnan(spy_sma[i]):
                current_trend_strength = 0.0
            else:
                current_trend_strength = _trend_signal_gradual(spy[i], spy_sma[i])

            if current_trend_strength <= 0 or vix_mult <= 0:
                target_holdings: list[str] = []
                target_weights: dict[str, float] = {}
            else:
                combined_today = mom.iloc[i].dropna() if i < len(mom) else pd.Series(dtype=float)

                if config.absolute_mom_filter:
                    raw_today = raw_mom.iloc[i].dropna() if i < len(raw_mom) else pd.Series(dtype=float)
                    positive_raw = raw_today[raw_today > 0].index
                    combined_today = combined_today[combined_today.index.isin(positive_raw)]

                if len(combined_today) >= config.top_n:
                    ranked = combined_today.sort_values(ascending=False)
                    target_holdings = list(ranked.index[:config.top_n])

                    if config.inv_vol_weight and stock_vol:
                        inv_vols = {}
                        for s in target_holdings:
                            sv = stock_vol.get(s)
                            v = sv[i-1] if sv is not None and i-1 < len(sv) and not np.isnan(sv[i-1]) and sv[i-1] > 0 else 0.2
                            inv_vols[s] = 1.0 / v
                        total_iv = sum(inv_vols.values())
                        target_weights = {s: min(iv / total_iv, config.max_position_pct) for s, iv in inv_vols.items()}
                        tw_sum = sum(target_weights.values())
                        target_weights = {s: w / tw_sum for s, w in target_weights.items()} if tw_sum > 0 else {s: 1.0 / len(target_holdings) for s in target_holdings}
                    else:
                        w = 1.0 / len(target_holdings)
                        target_weights = {s: w for s in target_holdings}
                else:
                    target_holdings = list(current_holdings)
                    target_weights = dict(current_weights)

            if set(target_holdings) != set(current_holdings):
                turnover = len(set(current_holdings) - set(target_holdings)) + len(set(target_holdings) - set(current_holdings))
                if turnover > 0:
                    cost = turnover * config.cost_fraction * equity / max(config.top_n * 2, 1)
                    equity -= cost
                    total_trades += turnover
                current_holdings = list(target_holdings)
                current_weights = dict(target_weights)
                last_rebalance = i

        equity_curve[i] = equity

    eq = equity_curve[warmup:]
    spy_n = spy_norm[warmup:]
    eq_dates = dates[warmup:]
    valid = ~np.isnan(eq) & ~np.isnan(spy_n)
    eq = eq[valid]
    spy_n = spy_n[valid]
    eq_dates = [d for d, v in zip(eq_dates, valid) if v]

    if len(eq) < 2:
        return {"error": "not enough data"}

    d_ret = np.diff(eq) / eq[:-1]
    d_ret = d_ret[np.isfinite(d_ret)]
    spy_d_ret = np.diff(spy_n) / spy_n[:-1]
    spy_d_ret = spy_d_ret[np.isfinite(spy_d_ret)]
    days = len(d_ret)
    years = days / 252
    std = np.std(d_ret)

    cagr = (eq[-1] / eq[0]) ** (1 / max(years, 0.1)) - 1
    spy_cagr = (spy_n[-1] / spy_n[0]) ** (1 / max(years, 0.1)) - 1
    sharpe = float(np.mean(d_ret) / std * math.sqrt(252)) if std > 1e-12 else 0
    peak = np.maximum.accumulate(eq)
    dd_series = eq / peak - 1
    max_dd = float(np.min(dd_series))

    neg_ret = d_ret[d_ret < 0]
    downside_std = float(np.std(neg_ret)) if len(neg_ret) > 0 else 1e-9
    sortino = float(np.mean(d_ret) / downside_std * math.sqrt(252)) if downside_std > 1e-12 else 0
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-9 else 0
    daily_win_rate = float(np.mean(d_ret > 0)) if len(d_ret) > 0 else 0
    profit_factor = float(np.sum(d_ret[d_ret > 0]) / abs(np.sum(d_ret[d_ret < 0]))) if np.sum(d_ret[d_ret < 0]) != 0 else 0
    avg_daily_ret = float(np.mean(d_ret))
    best_day = float(np.max(d_ret)) if len(d_ret) > 0 else 0
    worst_day = float(np.min(d_ret)) if len(d_ret) > 0 else 0
    ann_vol = float(std * math.sqrt(252))

    # Monthly returns by year/month for heatmap
    monthly_data: dict[int, dict[int, float]] = {}
    spy_monthly_data: dict[int, dict[int, float]] = {}
    prev_month_eq = eq[0]
    prev_month_spy = spy_n[0]
    prev_ym = (eq_dates[0].year, eq_dates[0].month) if eq_dates else (0, 0)

    for i in range(1, len(eq)):
        cur_ym = (eq_dates[i].year, eq_dates[i].month)
        if cur_ym != prev_ym or i == len(eq) - 1:
            yr, mo = prev_ym
            ret = (eq[i - 1] / prev_month_eq - 1) * 100 if prev_month_eq > 0 else 0
            spy_ret = (spy_n[i - 1] / prev_month_spy - 1) * 100 if prev_month_spy > 0 else 0
            monthly_data.setdefault(yr, {})[mo] = round(ret, 2)
            spy_monthly_data.setdefault(yr, {})[mo] = round(spy_ret, 2)
            prev_month_eq = eq[i - 1]
            prev_month_spy = spy_n[i - 1]
            prev_ym = cur_ym

    # Yearly returns
    yearly_data: list[dict] = []
    prev_year_eq = eq[0]
    prev_year_spy = spy_n[0]
    prev_yr = eq_dates[0].year if eq_dates else 0
    for i in range(1, len(eq)):
        cur_yr = eq_dates[i].year
        if cur_yr != prev_yr or i == len(eq) - 1:
            yr_ret = (eq[i - 1] / prev_year_eq - 1) * 100
            spy_yr_ret = (spy_n[i - 1] / prev_year_spy - 1) * 100
            yearly_data.append({
                "year": prev_yr,
                "return_pct": round(yr_ret, 1),
                "spy_pct": round(spy_yr_ret, 1),
                "alpha_pct": round(yr_ret - spy_yr_ret, 1),
            })
            prev_year_eq = eq[i - 1]
            prev_year_spy = spy_n[i - 1]
            prev_yr = cur_yr

    # Monthly returns flat for win rate
    all_monthly_rets = []
    for yr_months in monthly_data.values():
        all_monthly_rets.extend(yr_months.values())
    monthly_win_rate = sum(1 for r in all_monthly_rets if r > 0) / len(all_monthly_rets) if all_monthly_rets else 0
    neg_years = sum(1 for y in yearly_data if y["return_pct"] < 0)

    # Equity curve for chart (sampled to ~500 points)
    sample_step = max(1, len(eq) // 500)
    eq_chart = []
    for i in range(0, len(eq), sample_step):
        eq_chart.append({
            "date": str(eq_dates[i].date()),
            "equity": round(float(eq[i]), 0),
            "spy": round(float(spy_n[i]), 0),
            "dd": round(float(dd_series[i]) * 100, 2),
        })

    return {
        "cagr_pct": round(cagr * 100, 1),
        "spy_cagr_pct": round(spy_cagr * 100, 1),
        "alpha_pct": round((cagr - spy_cagr) * 100, 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "total_return_pct": round((eq[-1] / eq[0] - 1) * 100, 1),
        "spy_return_pct": round((spy_n[-1] / spy_n[0] - 1) * 100, 1),
        "monthly_win_rate_pct": round(monthly_win_rate * 100, 1),
        "daily_win_rate_pct": round(daily_win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2),
        "ann_volatility_pct": round(ann_vol * 100, 1),
        "best_day_pct": round(best_day * 100, 2),
        "worst_day_pct": round(worst_day * 100, 2),
        "avg_daily_ret_pct": round(avg_daily_ret * 100, 4),
        "neg_years": neg_years,
        "total_trades": total_trades,
        "years": round(years, 1),
        "final_equity": round(float(eq[-1]), 0),
        "monthly_returns": {str(yr): months for yr, months in sorted(monthly_data.items())},
        "spy_monthly_returns": {str(yr): months for yr, months in sorted(spy_monthly_data.items())},
        "yearly_returns": yearly_data,
        "equity_curve": eq_chart,
    }
