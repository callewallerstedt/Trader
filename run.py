#!/usr/bin/env python3
"""
Multi-Signal Momentum Rotation Bot V3 - Single entry point.

Signal: 0.5*momentum + 0.3*residual_momentum + 0.2*acceleration
Top-15 stocks, 25d rebalance, VIX filter (20/30), inv-vol weighted, no vol scaling.

Usage:
    python run.py download          Download 15+ years of daily data (+ VIX)
    python run.py backtest          Run backtest and show results
    python run.py signal            Show today's signal (what to buy/sell)
    python run.py trade             Dry run (compute orders, don't send)
    python run.py trade --live      Send MOC orders to IBKR for real
"""
from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "bot.log"),
    ],
)
log = logging.getLogger("trader")


def cmd_download():
    from strategy.data import download
    log.info("Downloading daily data from Yahoo Finance...")
    download()
    log.info("Download complete.")


def cmd_backtest():
    from strategy.data import load, load_vix
    from strategy.engine import Config, backtest

    log.info("Loading data...")
    daily = load()
    vix = load_vix()
    syms = sorted(daily["symbol"].unique())
    log.info(f"  {len(syms)} symbols, {daily['timestamp'].dt.date.nunique()} trading days")
    if vix is not None:
        log.info(f"  VIX data: {len(vix)} days")

    config = Config()
    m = backtest(daily, config, vix_series=vix)

    print(f"\n{'=' * 60}")
    print(f"  BACKTEST RESULTS  ({m['years']} years, {config.round_trip_cost_bps:.0f}bps cost)")
    print(f"{'=' * 60}")
    print(f"  Strategy CAGR:    {m['cagr_pct']:+.1f}%")
    print(f"  SPY CAGR:         {m['spy_cagr_pct']:+.1f}%")
    print(f"  Alpha:            {m['alpha_pct']:+.1f}%")
    print(f"  Sharpe:           {m['sharpe']:.2f}")
    print(f"  Sortino:          {m['sortino']:.2f}")
    print(f"  Max drawdown:     {m['max_drawdown_pct']:.1f}%")
    print(f"  Profit factor:    {m['profit_factor']:.2f}")
    print(f"  Monthly win rate: {m['monthly_win_rate_pct']:.0f}%")
    print(f"  Total return:     {m['total_return_pct']:+.1f}%  (SPY: {m['spy_return_pct']:+.1f}%)")
    print(f"  Final equity:     ${m['final_equity']:,.0f}")
    print(f"  Trades:           {m['total_trades']}")
    print(f"  Signal:           {config.sig_mom_weight:.0%} mom + "
          f"{config.sig_resid_weight:.0%} residual + "
          f"{config.sig_accel_weight:.0%} accel")
    print(f"  Config:           top-{config.top_n}, rebal {config.rebalance_freq}d, "
          f"inv-vol={'Y' if config.inv_vol_weight else 'N'}, "
          f"VIX {config.vix_reduce_threshold}/{config.vix_flat_threshold}")
    print(f"{'=' * 60}")


def cmd_signal():
    from strategy.data import fetch_live, fetch_vix_live
    from strategy.engine import Config, compute_signal

    log.info("Fetching live prices...")
    daily = fetch_live()
    vix_val = fetch_vix_live()
    signal = compute_signal(daily, Config(), vix_value=vix_val)

    print(f"\nDate:       {signal.get('date', '?')}")
    print(f"SPY:        ${signal.get('spy_price', 0):.2f}")
    print(f"SMA(200):   ${signal.get('spy_sma', 0):.2f}")
    print(f"SPY vs SMA: {signal.get('spy_pct_vs_sma', 0):+.2f}%")
    print(f"Trend:      {signal.get('trend', '?')} (strength: {signal.get('trend_strength', 0):.0f}%)")
    print(f"Exposure:   {signal.get('effective_exposure', signal.get('exposure', 1)):.1%}")
    if vix_val is not None:
        print(f"VIX:        {vix_val:.1f} (multiplier: {signal.get('vix_multiplier', 1):.1%})")
    print(f"Action:     {signal.get('action', '?')}")
    print(f"Strategy:   {signal.get('strategy', '?')}")
    if signal.get("target_holdings"):
        print(f"Hold:       {', '.join(signal['target_holdings'])}")
    if signal.get("target_weights"):
        print(f"\nTarget weights:")
        for sym, w in sorted(signal["target_weights"].items(), key=lambda x: -x[1]):
            print(f"  {sym:6s} {w:.1%}")
    if signal.get("momentum_scores"):
        print(f"\nTop 15 by momentum:")
        for sym, score in signal["momentum_scores"].items():
            print(f"  {sym:6s} {score:+.1f}%")


def _validate_signal_freshness(signal: dict) -> bool:
    """Ensure signal is based on recent data, not stale."""
    sig_date = signal.get("date", "")
    if not sig_date:
        return False
    from datetime import date
    sig = date.fromisoformat(sig_date)
    today = date.today()
    delta = (today - sig).days
    if delta > 5:
        log.warning(f"Signal date {sig_date} is {delta} days old - data may be stale")
        return False
    return True


def cmd_trade(live: bool = False):
    from strategy.data import fetch_live, fetch_vix_live
    from strategy.engine import Config, compute_signal
    from broker.ibkr import IBKRBroker, OrderRequest

    config = Config()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    trade_log: dict = {
        "run_id": ts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "live": live,
        "steps": [],
        "errors": [],
    }

    def step(msg: str, data: dict | None = None):
        log.info(msg)
        entry: dict = {"time": datetime.now(timezone.utc).isoformat(), "msg": msg}
        if data:
            entry["data"] = data
        trade_log["steps"].append(entry)

    def error(msg: str, data: dict | None = None):
        log.error(msg)
        entry: dict = {"time": datetime.now(timezone.utc).isoformat(), "msg": msg}
        if data:
            entry["data"] = data
        trade_log["errors"].append(entry)

    try:
        _write_status("fetching", "Fetching live prices from Yahoo Finance...")
        step("Fetching live prices from Yahoo Finance")
        daily = fetch_live()
        n_syms = daily["symbol"].nunique()
        n_rows = len(daily)
        step(f"Fetched {n_rows} rows for {n_syms} symbols")

        if n_syms < 15:
            error(f"Only {n_syms} symbols fetched - data quality issue")

        vix_val = fetch_vix_live()
        step(f"VIX: {vix_val:.1f}" if vix_val else "VIX: unavailable")

        _write_status("computing", "Computing signal...")
        step("Computing signal")
        signal = compute_signal(daily, config, vix_value=vix_val)
        trade_log["signal"] = {k: v for k, v in signal.items() if k != "spy_history"}

        if not _validate_signal_freshness(signal):
            error(f"Signal based on stale data: {signal.get('date', '?')}")

        step(f"Signal: action={signal.get('action')} trend={signal.get('trend')} "
             f"strength={signal.get('trend_strength', 0):.0f}% "
             f"SPY=${signal.get('spy_price', 0):.2f} SMA=${signal.get('spy_sma', 0):.2f} "
             f"exposure={signal.get('exposure', 1):.1%} "
             f"VIX mult={signal.get('vix_multiplier', 1):.0%}")

        if signal["action"] not in ("hold", "go_to_cash"):
            step(f"No action needed: {signal.get('reason', '?')}")
            _save_trade_log(trade_log, ts)
            return

        target = set(signal.get("target_holdings", []))
        target_weights = signal.get("target_weights", {})
        exposure = signal.get("effective_exposure", signal.get("exposure", 1.0))

        broker = IBKRBroker()
        current_pos: dict[str, float] = {}
        equity = 100_000.0
        currency = "USD"

        if live:
            _write_status("connecting", "Connecting to IBKR Gateway...")
            step("Connecting to IBKR Gateway")
            if not broker.connect():
                error("Could not connect to IBKR Gateway after retries")
                _write_status("error", "Could not connect to IBKR Gateway")
                _save_trade_log(trade_log, ts)
                return

            cancelled = broker.cancel_all_orders()
            if cancelled:
                step(f"Cancelled {cancelled} stale open orders")

            positions = broker.get_positions()
            current_pos = {p.symbol: p.quantity for p in positions}
            equity = broker.get_equity() or 100_000.0
            currency = broker.get_currency()
            step(f"IBKR connected: equity={equity:,.2f} {currency}, "
                 f"positions={current_pos or 'none'}")

            # Convert equity to USD (stock prices are in USD)
            if currency.upper() != "USD":
                fx_rate = broker.get_fx_rate("USD")
                if fx_rate <= 0:
                    error(f"Could not get {currency}->USD exchange rate from IBKR, aborting")
                    broker.disconnect()
                    _save_trade_log(trade_log, ts)
                    return
                equity_usd = equity / fx_rate
                step(f"FX conversion: {equity:,.2f} {currency} / {fx_rate:.4f} = "
                     f"${equity_usd:,.2f} USD")
                equity = equity_usd

            # Paper trading simulation: proportional cap that grows with profits
            paper_sim = _load_paper_sim()
            if paper_sim:
                baseline_sek = paper_sim.get("ibkr_baseline_sek", 0)
                sim_sek = paper_sim.get("sim_sek", 0)
                if baseline_sek > 0 and sim_sek > 0:
                    ibkr_equity_sek = equity * fx_rate  # back to SEK
                    sim_equity_sek = ibkr_equity_sek * (sim_sek / baseline_sek)
                    sim_equity_usd = sim_equity_sek / fx_rate
                    step(f"Paper sim: {sim_equity_sek:,.0f} SEK (~${sim_equity_usd:,.0f} USD) "
                         f"[{sim_sek:,}/{baseline_sek:,.0f} SEK ratio, grows with P&L]")
                    equity = sim_equity_usd

            trade_log["account"] = {
                "equity": equity,
                "currency": "USD",
                "positions": current_pos,
            }
        else:
            step("DRY RUN mode - not connecting to IBKR")

        orders: list[OrderRequest] = []
        current = set(current_pos.keys())

        # Sell positions no longer in target
        for sym in current - target:
            qty = abs(int(current_pos[sym]))
            if qty > 0:
                orders.append(OrderRequest(sym, "SELL", qty))
                step(f"Order: SELL {qty} {sym} (exit position)")

        # Size target positions using inverse-vol weights
        if target:
            scaled_equity = equity * exposure
            prices = signal.get("prices", {})
            step(f"Position sizing: equity={equity:,.0f} x exposure={exposure:.1%} "
                 f"= {scaled_equity:,.0f} across {len(target)} stocks")

            for sym in target:
                price = prices.get(sym, 0)
                if price <= 0:
                    error(f"No price for {sym}, skipping")
                    continue

                weight = target_weights.get(sym, 1.0 / len(target))
                alloc = scaled_equity * weight
                desired_qty = int(alloc / price)
                if desired_qty <= 0:
                    step(f"Skip {sym}: alloc ${alloc:.0f} < 1 share @ ${price:.2f} (weight={weight:.1%})")
                    continue

                current_qty = int(current_pos.get(sym, 0))
                delta = desired_qty - current_qty

                if delta > 0:
                    orders.append(OrderRequest(sym, "BUY", delta))
                    step(f"Order: BUY {delta} {sym} @ ~${price:.2f} "
                         f"(have {current_qty}, want {desired_qty}, weight={weight:.1%})")
                elif delta < 0:
                    orders.append(OrderRequest(sym, "SELL", abs(delta)))
                    step(f"Order: SELL {abs(delta)} {sym} "
                         f"(rebalance: have {current_qty}, want {desired_qty}, weight={weight:.1%})")
                else:
                    step(f"No change for {sym} (already holding {current_qty})")

        trade_log["orders_planned"] = [
            {"symbol": o.symbol, "side": o.side, "quantity": o.quantity, "type": o.order_type}
            for o in orders
        ]

        if not orders:
            step("No trades needed - portfolio already matches target")
            _write_status("complete", "No trades needed — portfolio matches target",
                          {"orders": 0, "action": signal.get("action")})
        elif live:
            _write_status("submitting", f"Submitting {len(orders)} MOC orders to IBKR...")
            step(f"Submitting {len(orders)} orders to IBKR (sells first, then buys)")
            results = broker.submit_orders_batch(orders)

            for r in results:
                status = r.get("status", "unknown")
                sym = r.get("symbol", "?")
                side = r.get("side", "?")
                qty = r.get("quantity", 0)
                filled = r.get("filled", 0)
                avg_px = r.get("avg_price", 0)
                step(f"  {side} {qty} {sym}: status={status} filled={filled} "
                     f"avg_price={avg_px}", r)

                if status in ("ApiError", "Cancelled", "Inactive", "exception",
                              "not_connected", "qualify_failed"):
                    error(f"Order failed: {side} {qty} {sym} -> {status}", r)

            trade_log["order_results"] = results

            _write_status("moc_pending", f"{len(orders)} MOC orders placed — waiting for 4 PM close",
                          {"orders": len(orders), "action": signal.get("action"),
                           "symbols": [o.symbol for o in orders]})
            step("Waiting 5s for fill updates")
            broker._ib.sleep(5)

            final_positions = broker.get_positions()
            final_equity = broker.get_equity() or equity
            step(f"Post-trade: equity={final_equity:,.2f} {currency}")
            trade_log["post_trade"] = {
                "equity": final_equity,
                "positions": {
                    p.symbol: {"qty": p.quantity, "cost": p.avg_cost}
                    for p in final_positions
                },
            }
            broker.disconnect()
            step("Disconnected from IBKR")
            _write_status("complete", f"Trade complete — {len(orders)} orders placed",
                          {"orders": len(orders), "action": signal.get("action"),
                           "symbols": [o.symbol for o in orders]})
        else:
            step(f"DRY RUN - {len(orders)} orders NOT sent. Use --live to execute.")

    except Exception as e:
        error(f"Unhandled exception: {e}", {"traceback": traceback.format_exc()})
        log.exception("Trade run failed")
        _write_status("error", f"Unhandled exception: {e}")

    _save_trade_log(trade_log, ts)


def _save_trade_log(trade_log: dict, ts: str):
    path = LOG_DIR / f"{ts}.json"
    path.write_text(json.dumps(trade_log, indent=2, default=str))
    log.info(f"Trade log saved: {path}")


STATUS_FILE = Path("status.json")

def _write_status(state: str, phase: str, extra: dict | None = None):
    data = {
        "state": state,
        "phase": phase,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        data.update(extra)
    try:
        STATUS_FILE.write_text(json.dumps(data, indent=2, default=str))
    except Exception:
        pass


def _load_paper_sim() -> dict | None:
    """Load paper simulation config (proportional equity cap)."""
    p = Path("paper_sim.json")
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "download":
        cmd_download()
    elif cmd == "backtest":
        cmd_backtest()
    elif cmd == "signal":
        cmd_signal()
    elif cmd == "trade":
        live = "--live" in sys.argv
        cmd_trade(live=live)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
