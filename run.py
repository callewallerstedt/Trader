#!/usr/bin/env python3
"""
Multi-TF Momentum Rotation Bot - Single entry point.

Usage:
    python run.py download          Download 15+ years of daily data
    python run.py backtest          Run backtest and show results
    python run.py signal            Show today's signal (what to buy/sell)
    python run.py trade             Dry run (compute orders, don't send)
    python run.py trade --live      Send MOC orders to IBKR for real
"""
from __future__ import annotations

import json
import logging
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
    from strategy.data import load
    from strategy.engine import Config, backtest

    log.info("Loading data...")
    daily = load()
    syms = sorted(daily["symbol"].unique())
    log.info(f"  {len(syms)} symbols, {daily['timestamp'].dt.date.nunique()} trading days")

    config = Config()
    m = backtest(daily, config)

    print(f"\n{'=' * 55}")
    print(f"  BACKTEST RESULTS  ({m['years']} years, 25bps cost)")
    print(f"{'=' * 55}")
    print(f"  Strategy CAGR:    {m['cagr_pct']:+.1f}%")
    print(f"  SPY CAGR:         {m['spy_cagr_pct']:+.1f}%")
    print(f"  Alpha:            {m['alpha_pct']:+.1f}%")
    print(f"  Sharpe:           {m['sharpe']:.2f}")
    print(f"  Max drawdown:     {m['max_drawdown_pct']:.1f}%")
    print(f"  Monthly win rate: {m['monthly_win_rate_pct']:.0f}%")
    print(f"  Total return:     {m['total_return_pct']:+.1f}%  (SPY: {m['spy_return_pct']:+.1f}%)")
    print(f"  Final equity:     ${m['final_equity']:,.0f}")
    print(f"  Trades:           {m['total_trades']}")
    print(f"{'=' * 55}")


def cmd_signal():
    from strategy.data import fetch_live
    from strategy.engine import Config, compute_signal

    log.info("Fetching live prices...")
    daily = fetch_live()
    signal = compute_signal(daily, Config())

    print(f"\nDate:      {signal.get('date', '?')}")
    print(f"SPY:       ${signal.get('spy_price', 0):.2f}")
    print(f"SMA(200):  ${signal.get('spy_sma', 0):.2f}")
    print(f"SPY vs SMA:{signal.get('spy_pct_vs_sma', 0):+.2f}%")
    print(f"Trend:     {signal.get('trend', '?')} (strength: {signal.get('trend_strength', 0):.0f}%)")
    print(f"Exposure:  {signal.get('exposure', 1):.1%}")
    print(f"Vol (ann): {signal.get('realized_vol', 0):.1f}%")
    print(f"Action:    {signal.get('action', '?')}")
    print(f"Strategy:  {signal.get('strategy', '?')}")
    if signal.get("target_holdings"):
        print(f"Hold:      {', '.join(signal['target_holdings'])}")
    if signal.get("momentum_scores"):
        print(f"\nTop 10 by momentum:")
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
    from strategy.data import fetch_live
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
        step("Fetching live prices from Yahoo Finance")
        daily = fetch_live()
        n_syms = daily["symbol"].nunique()
        n_rows = len(daily)
        step(f"Fetched {n_rows} rows for {n_syms} symbols")

        if n_syms < 10:
            error(f"Only {n_syms} symbols fetched - data quality issue")

        step("Computing signal")
        signal = compute_signal(daily, config)
        trade_log["signal"] = {k: v for k, v in signal.items() if k != "spy_history"}

        if not _validate_signal_freshness(signal):
            error(f"Signal based on stale data: {signal.get('date', '?')}")

        step(f"Signal: action={signal.get('action')} trend={signal.get('trend')} "
             f"strength={signal.get('trend_strength', 0):.0f}% "
             f"SPY=${signal.get('spy_price', 0):.2f} SMA=${signal.get('spy_sma', 0):.2f} "
             f"exposure={signal.get('exposure', 1):.1%}")

        if signal["action"] not in ("hold", "go_to_cash"):
            step(f"No action needed: {signal.get('reason', '?')}")
            _save_trade_log(trade_log, ts)
            return

        target = set(signal.get("target_holdings", []))
        exposure = signal.get("effective_exposure", signal.get("exposure", 1.0))

        broker = IBKRBroker()
        current_pos: dict[str, float] = {}
        equity = 100_000.0
        currency = "USD"

        if live:
            step("Connecting to IBKR Gateway")
            if not broker.connect():
                error("Could not connect to IBKR Gateway after retries")
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
            trade_log["account"] = {
                "equity": equity,
                "currency": currency,
                "positions": current_pos,
            }
        else:
            step("DRY RUN mode - not connecting to IBKR")

        orders: list[OrderRequest] = []
        current = set(current_pos.keys())

        for sym in current - target:
            qty = abs(int(current_pos[sym]))
            if qty > 0:
                orders.append(OrderRequest(sym, "SELL", qty))
                step(f"Order: SELL {qty} {sym} (exit position)")

        if target:
            scaled_equity = equity * exposure
            alloc_per_stock = scaled_equity / len(target)
            prices = signal.get("prices", {})
            step(f"Position sizing: equity={equity:,.0f} x exposure={exposure:.1%} "
                 f"= {scaled_equity:,.0f}, per-stock={alloc_per_stock:,.0f}")

            for sym in target:
                price = prices.get(sym, 0)
                if price <= 0:
                    error(f"No price for {sym}, skipping")
                    continue

                desired_qty = int(alloc_per_stock / price)
                if desired_qty <= 0:
                    error(f"Computed 0 shares for {sym} (alloc={alloc_per_stock:.0f}, price={price:.2f})")
                    continue

                current_qty = int(current_pos.get(sym, 0))
                delta = desired_qty - current_qty

                if delta > 0:
                    orders.append(OrderRequest(sym, "BUY", delta))
                    step(f"Order: BUY {delta} {sym} @ ~${price:.2f} "
                         f"(have {current_qty}, want {desired_qty})")
                elif delta < 0:
                    orders.append(OrderRequest(sym, "SELL", abs(delta)))
                    step(f"Order: SELL {abs(delta)} {sym} "
                         f"(rebalance: have {current_qty}, want {desired_qty})")
                else:
                    step(f"No change for {sym} (already holding {current_qty})")

        trade_log["orders_planned"] = [
            {"symbol": o.symbol, "side": o.side, "quantity": o.quantity, "type": o.order_type}
            for o in orders
        ]

        if not orders:
            step("No trades needed - portfolio already matches target")
        elif live:
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
        else:
            step(f"DRY RUN - {len(orders)} orders NOT sent. Use --live to execute.")

    except Exception as e:
        error(f"Unhandled exception: {e}", {"traceback": traceback.format_exc()})
        log.exception("Trade run failed")

    _save_trade_log(trade_log, ts)


def _save_trade_log(trade_log: dict, ts: str):
    path = LOG_DIR / f"{ts}.json"
    path.write_text(json.dumps(trade_log, indent=2, default=str))
    log.info(f"Trade log saved: {path}")


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
