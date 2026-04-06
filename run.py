#!/usr/bin/env python3
"""
Momentum Rotation Trading Bot - Single entry point.

Usage:
    python run.py download          Download 15 years of daily data
    python run.py backtest          Run backtest and show results
    python run.py signal            Show today's signal (what to buy/sell)
    python run.py trade             Dry run (compute orders, don't send)
    python run.py trade --live      Send MOC orders to IBKR for real
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


def cmd_download():
    from strategy.data import download
    print("Downloading daily data from Yahoo Finance...")
    download()
    print("Done.")


def cmd_backtest():
    from strategy.data import load
    from strategy.engine import Config, backtest

    print("Loading data...")
    daily = load()
    syms = sorted(daily["symbol"].unique())
    print(f"  {len(syms)} symbols, {daily['timestamp'].dt.date.nunique()} trading days")

    config = Config()
    m = backtest(daily, config)

    print(f"\n{'=' * 50}")
    print(f"  BACKTEST RESULTS  ({m['years']} years)")
    print(f"{'=' * 50}")
    print(f"  Strategy CAGR:    {m['cagr_pct']:+.1f}%")
    print(f"  SPY CAGR:         {m['spy_cagr_pct']:+.1f}%")
    print(f"  Alpha:            {m['alpha_pct']:+.1f}%")
    print(f"  Sharpe:           {m['sharpe']:.2f}")
    print(f"  Max drawdown:     {m['max_drawdown_pct']:.1f}%")
    print(f"  Total return:     {m['total_return_pct']:+.1f}%  (SPY: {m['spy_return_pct']:+.1f}%)")
    print(f"  Final equity:     ${m['final_equity']:,.0f}")
    print(f"  Trades:           {m['total_trades']}")
    print(f"{'=' * 50}")


def cmd_signal():
    from strategy.data import fetch_live
    from strategy.engine import Config, compute_signal

    print("Fetching live prices...")
    daily = fetch_live()
    signal = compute_signal(daily, Config())

    print(f"\nDate:    {signal.get('date', '?')}")
    print(f"SPY:     ${signal.get('spy_price', 0):.2f}")
    print(f"SMA:     ${signal.get('spy_sma', 0):.2f}")
    print(f"Trend:   {signal.get('trend', '?')}")
    print(f"Action:  {signal.get('action', '?')}")
    if signal.get("target_holdings"):
        print(f"Hold:    {', '.join(signal['target_holdings'])}")
    if signal.get("momentum_scores"):
        print(f"\nTop 10 by momentum:")
        for sym, score in signal["momentum_scores"].items():
            print(f"  {sym:6s} {score:+.1f}%")


def cmd_trade(live: bool = False):
    from strategy.data import fetch_live
    from strategy.engine import Config, compute_signal
    from broker.ibkr import IBKRBroker, OrderRequest

    config = Config()
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 1. Fetch prices and compute signal
    print("Fetching live prices...")
    daily = fetch_live()
    signal = compute_signal(daily, config)

    print(f"[{signal.get('date', '?')}] SPY=${signal.get('spy_price', 0):.2f} "
          f"SMA=${signal.get('spy_sma', 0):.2f} Trend={signal.get('trend', '?')}")
    print(f"  Action: {signal.get('action', '?')}")

    if signal["action"] not in ("hold", "go_to_cash"):
        print(f"  Reason: {signal.get('reason', '?')}")
        return

    target = set(signal.get("target_holdings", []))

    # 2. Connect to IBKR (only if live)
    broker = IBKRBroker()
    current_pos: dict[str, float] = {}
    equity = 100_000.0

    if live:
        if not broker.connect():
            print("  ERROR: Could not connect to IBKR. Aborting.")
            return
        positions = broker.get_positions()
        current_pos = {p.symbol: p.quantity for p in positions}
        equity = broker.get_equity() or 100_000.0
        print(f"  IBKR equity: ${equity:,.0f}")
        print(f"  Current positions: {current_pos or 'none'}")

    # 3. Generate orders
    orders: list[OrderRequest] = []
    current = set(current_pos.keys())

    for sym in current - target:
        qty = abs(int(current_pos[sym]))
        if qty > 0:
            orders.append(OrderRequest(sym, "SELL", qty))

    if target:
        alloc = equity / len(target)
        prices = signal.get("prices", {})
        for sym in target - current:
            price = prices.get(sym, 0)
            if price > 0:
                qty = int(alloc / price)
                if qty > 0:
                    orders.append(OrderRequest(sym, "BUY", qty))

    # 4. Execute or print
    if not orders:
        print("  No trades needed.")
    else:
        for o in orders:
            print(f"  Order: {o.side} {o.quantity} {o.symbol} ({o.order_type})")

    if live and orders:
        print("\n  Submitting orders to IBKR...")
        for o in orders:
            result = broker.submit_order(o)
            print(f"    {o.symbol}: {result.get('status', '?')}")
        broker.disconnect()
    elif not live:
        print(f"\n  DRY RUN - orders not sent. Use 'python run.py trade --live' to execute.")

    # 5. Save log
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "live": live,
        "signal": signal,
        "orders": [o.__dict__ for o in orders],
        "equity": equity,
    }
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    (log_dir / f"{ts}.json").write_text(json.dumps(log_entry, indent=2, default=str))


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
