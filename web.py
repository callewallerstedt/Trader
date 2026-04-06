#!/usr/bin/env python3
"""
Live monitoring dashboard for the Momentum Rotation Bot.
Run: python web.py
Then open http://YOUR_IP:8080
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

DATA_DIR = Path("data")
LOG_DIR = Path("logs")


def _get_signal() -> dict:
    """Compute live signal from current prices."""
    try:
        from strategy.data import fetch_live
        from strategy.engine import Config, compute_signal
        daily = fetch_live()
        return compute_signal(daily, Config())
    except Exception as e:
        return {"action": "error", "reason": str(e)}


def _get_backtest() -> dict:
    """Run backtest and return metrics."""
    try:
        from strategy.data import load
        from strategy.engine import Config, backtest
        daily = load()
        return backtest(daily, Config())
    except Exception as e:
        return {"error": str(e)}


def _get_trade_logs() -> list[dict]:
    """Read recent trade logs."""
    logs = []
    if not LOG_DIR.exists():
        return logs
    for f in sorted(LOG_DIR.glob("*.json"), reverse=True)[:30]:
        try:
            data = json.loads(f.read_text())
            data["_file"] = f.name
            logs.append(data)
        except Exception:
            continue
    return logs


def _get_cron_log(lines: int = 40) -> str:
    """Read tail of trade.log."""
    log_file = Path("trade.log")
    if not log_file.exists():
        return "No trade.log yet. Bot hasn't run via cron yet."
    text = log_file.read_text()
    return "\n".join(text.strip().split("\n")[-lines:])


_signal_cache: dict[str, Any] = {"data": None, "time": None}
_backtest_cache: dict[str, Any] = {"data": None, "time": None}


def _cached_signal() -> dict:
    now = datetime.now()
    if _signal_cache["data"] and _signal_cache["time"] and (now - _signal_cache["time"]).seconds < 120:
        return _signal_cache["data"]
    sig = _get_signal()
    _signal_cache["data"] = sig
    _signal_cache["time"] = now
    return sig


def _cached_backtest() -> dict:
    if _backtest_cache["data"]:
        return _backtest_cache["data"]
    bt = _get_backtest()
    _backtest_cache["data"] = bt
    _backtest_cache["time"] = datetime.now()
    return bt


@app.get("/api/signal")
def api_signal():
    return _cached_signal()


@app.get("/api/backtest")
def api_backtest():
    return _cached_backtest()


@app.get("/api/logs")
def api_logs():
    return _get_trade_logs()


@app.get("/api/cronlog")
def api_cronlog():
    return {"log": _get_cron_log()}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    sig = _cached_signal()
    bt = _cached_backtest()
    logs = _get_trade_logs()
    cron_log = _get_cron_log(25)
    now = datetime.now()

    action = sig.get("action", "unknown")
    trend = sig.get("trend", "?")
    spy_price = sig.get("spy_price", 0)
    spy_sma = sig.get("spy_sma", 0)
    holdings = sig.get("target_holdings", [])
    mom_scores = sig.get("momentum_scores", {})
    sig_date = sig.get("date", "?")

    if action == "hold":
        status_color = "#22c55e"
        status_text = "HOLDING"
        status_detail = ", ".join(holdings)
    elif action == "go_to_cash":
        status_color = "#f59e0b"
        status_text = "CASH"
        status_detail = sig.get("reason", "SPY below SMA")
    elif action == "wait":
        status_color = "#6b7280"
        status_text = "WAITING"
        status_detail = sig.get("reason", "")
    else:
        status_color = "#ef4444"
        status_text = "ERROR"
        status_detail = sig.get("reason", str(sig))

    # Next run time
    today = now.replace(hour=15, minute=45, second=0, microsecond=0)
    if now > today or now.weekday() >= 5:
        days_ahead = 1
        next_run = today + timedelta(days=days_ahead)
        while next_run.weekday() >= 5:
            next_run += timedelta(days=1)
    else:
        next_run = today
    next_run_str = next_run.strftime("%A %b %d, %I:%M %p ET")
    time_until = next_run - now
    hours_until = int(time_until.total_seconds() // 3600)
    mins_until = int((time_until.total_seconds() % 3600) // 60)

    # Momentum table
    mom_rows = ""
    for i, (sym, score) in enumerate(mom_scores.items()):
        is_held = sym in holdings
        badge = '<span class="badge">HOLD</span>' if is_held else ""
        bar_w = min(max(score, 0), 60)
        bar_color = "#22c55e" if score > 0 else "#ef4444"
        mom_rows += f"""<tr class="{'held' if is_held else ''}">
            <td>{i+1}</td><td>{sym} {badge}</td>
            <td><div class="bar" style="width:{bar_w}%;background:{bar_color}"></div></td>
            <td class="num">{score:+.1f}%</td></tr>"""

    # Trade log rows
    log_rows = ""
    for log in logs[:15]:
        ts = log.get("timestamp", "?")[:16]
        live = "LIVE" if log.get("live") else "DRY"
        sig_a = log.get("signal", {}).get("action", "?")
        orders = log.get("orders", [])
        order_str = ", ".join(f"{o['side']} {o['quantity']} {o['symbol']}" for o in orders) if orders else "no trades"
        log_rows += f"<tr><td>{ts}</td><td><span class='tag {live.lower()}'>{live}</span></td><td>{sig_a}</td><td>{order_str}</td></tr>"

    # Backtest stats
    bt_cagr = bt.get("cagr_pct", 0)
    bt_spy = bt.get("spy_cagr_pct", 0)
    bt_alpha = bt.get("alpha_pct", 0)
    bt_sharpe = bt.get("sharpe", 0)
    bt_dd = bt.get("max_drawdown_pct", 0)
    bt_years = bt.get("years", 0)

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Momentum Bot</title>
<meta http-equiv="refresh" content="120">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0a0a0a;color:#e5e5e5;padding:20px;max-width:1100px;margin:0 auto}}
h1{{font-size:1.4rem;font-weight:600;margin-bottom:4px;color:#fff}}
.subtitle{{color:#737373;font-size:.85rem;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px}}
.card{{background:#171717;border:1px solid #262626;border-radius:10px;padding:20px}}
.card h2{{font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;color:#737373;margin-bottom:10px}}
.status-box{{text-align:center;padding:28px 20px}}
.status-dot{{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:8px;animation:pulse 2s infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.4}}}}
.status-label{{font-size:1.8rem;font-weight:700;color:{status_color}}}
.status-detail{{color:#a3a3a3;margin-top:6px;font-size:.95rem}}
.kpi{{text-align:center}}
.kpi .val{{font-size:1.6rem;font-weight:700;color:#fff}}
.kpi .label{{font-size:.75rem;color:#737373;margin-top:2px}}
.kpi-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:12px}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th{{text-align:left;color:#737373;font-weight:500;padding:8px 10px;border-bottom:1px solid #262626;font-size:.75rem;text-transform:uppercase}}
td{{padding:7px 10px;border-bottom:1px solid #1a1a1a}}
tr.held td{{background:#052e16}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.bar{{height:6px;border-radius:3px;min-width:2px}}
.badge{{background:#166534;color:#4ade80;font-size:.65rem;padding:2px 6px;border-radius:4px;margin-left:6px;font-weight:600}}
.tag{{font-size:.7rem;padding:2px 8px;border-radius:4px;font-weight:600}}
.tag.live{{background:#7c2d12;color:#fb923c}}
.tag.dry{{background:#1e3a5f;color:#60a5fa}}
.next-run{{background:#172554;border-color:#1e3a5f}}
.next-run .val{{color:#60a5fa}}
pre{{background:#111;padding:12px;border-radius:6px;font-size:.78rem;overflow-x:auto;max-height:280px;overflow-y:auto;color:#a3a3a3;line-height:1.5}}
.refresh{{color:#525252;font-size:.75rem;margin-top:16px;text-align:center}}
.refresh a{{color:#525252}}
.spy-info{{display:flex;gap:20px;justify-content:center;margin-top:10px}}
.spy-info span{{font-size:.85rem}}
.spy-info .lbl{{color:#737373}}
.full{{grid-column:1/-1}}
</style></head><body>

<h1>Momentum Rotation Bot</h1>
<p class="subtitle">Live monitoring &middot; auto-refreshes every 2 min &middot; {now.strftime('%b %d %Y, %I:%M %p ET')}</p>

<div class="grid">

<div class="card status-box">
<h2>Current Signal</h2>
<div><span class="status-dot" style="background:{status_color}"></span><span class="status-label">{status_text}</span></div>
<div class="status-detail">{status_detail}</div>
<div class="spy-info">
<span><span class="lbl">SPY</span> ${spy_price:.2f}</span>
<span><span class="lbl">SMA(100)</span> ${spy_sma:.2f}</span>
<span><span class="lbl">Trend</span> {trend}</span>
</div>
<div style="color:#525252;font-size:.75rem;margin-top:8px">Signal as of {sig_date} close prices</div>
</div>

<div class="card next-run">
<h2>Next Run</h2>
<div class="kpi"><div class="val">{next_run_str}</div><div class="label">in {hours_until}h {mins_until}m</div></div>
<div class="kpi-grid" style="margin-top:18px">
<div class="kpi"><div class="val" style="font-size:1.1rem">{bt_cagr:+.1f}%</div><div class="label">Strategy CAGR</div></div>
<div class="kpi"><div class="val" style="font-size:1.1rem">{bt_spy:+.1f}%</div><div class="label">SPY CAGR</div></div>
<div class="kpi"><div class="val" style="font-size:1.1rem;color:#22c55e">{bt_alpha:+.1f}%</div><div class="label">Alpha</div></div>
</div>
<div class="kpi-grid">
<div class="kpi"><div class="val" style="font-size:1.1rem">{bt_sharpe:.2f}</div><div class="label">Sharpe</div></div>
<div class="kpi"><div class="val" style="font-size:1.1rem;color:#ef4444">{bt_dd:.1f}%</div><div class="label">Max DD</div></div>
<div class="kpi"><div class="val" style="font-size:1.1rem">{bt_years}</div><div class="label">Years tested</div></div>
</div>
</div>

<div class="card">
<h2>Momentum Rankings (20-day)</h2>
<table><thead><tr><th>#</th><th>Symbol</th><th>Momentum</th><th class="num">Return</th></tr></thead>
<tbody>{mom_rows if mom_rows else '<tr><td colspan="4" style="color:#525252">SPY below SMA &mdash; no rankings while in cash</td></tr>'}</tbody></table>
</div>

<div class="card">
<h2>Trade Log</h2>
<table><thead><tr><th>Time</th><th>Mode</th><th>Signal</th><th>Orders</th></tr></thead>
<tbody>{log_rows if log_rows else '<tr><td colspan="4" style="color:#525252">No trades yet. Bot will log here after first run.</td></tr>'}</tbody></table>
</div>

<div class="card full">
<h2>Cron Log (trade.log)</h2>
<pre>{cron_log}</pre>
</div>

</div>

<div class="refresh">Auto-refreshes every 2 minutes &middot; <a href="/">Refresh now</a> &middot; <a href="/api/signal">API: /api/signal</a></div>

</body></html>"""
    return html


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Dashboard starting at http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
