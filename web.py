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
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

DATA_DIR = Path("data")
LOG_DIR = Path("logs")


def _get_signal() -> dict:
    try:
        from strategy.data import fetch_live
        from strategy.engine import Config, compute_signal
        daily = fetch_live()
        return compute_signal(daily, Config())
    except Exception as e:
        return {"action": "error", "reason": str(e)}


def _get_backtest() -> dict:
    try:
        from strategy.data import load
        from strategy.engine import Config, backtest
        daily = load()
        return backtest(daily, Config())
    except Exception as e:
        return {"error": str(e)}


def _get_trade_logs() -> list[dict]:
    logs = []
    if not LOG_DIR.exists():
        return logs
    for f in sorted(LOG_DIR.glob("*.json"), reverse=True)[:50]:
        try:
            data = json.loads(f.read_text())
            data["_file"] = f.name
            logs.append(data)
        except Exception:
            continue
    return logs


def _get_cron_log(lines: int = 50) -> str:
    log_file = Path("trade.log")
    if not log_file.exists():
        return "No trade.log yet."
    text = log_file.read_text()
    return "\n".join(text.strip().split("\n")[-lines:])


def _get_bot_log(lines: int = 60) -> str:
    log_file = LOG_DIR / "bot.log"
    if not log_file.exists():
        return "No bot.log yet."
    text = log_file.read_text()
    return "\n".join(text.strip().split("\n")[-lines:])


def _get_ibkr_status() -> dict:
    try:
        from broker.ibkr import IBKRBroker
        broker = IBKRBroker(client_id=50)
        if not broker.connect():
            return {"connected": False}
        equity = broker.get_equity()
        currency = broker.get_currency()
        positions = broker.get_positions()
        summary = broker.get_account_summary()
        broker.disconnect()
        return {
            "connected": True,
            "equity": equity,
            "currency": currency,
            "positions": [
                {"symbol": p.symbol, "qty": p.quantity, "avg_cost": p.avg_cost}
                for p in positions
            ],
            "summary": summary,
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


_signal_cache: dict[str, Any] = {"data": None, "time": None}
_backtest_cache: dict[str, Any] = {"data": None, "time": None}
_ibkr_cache: dict[str, Any] = {"data": None, "time": None}


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


def _cached_ibkr() -> dict:
    now = datetime.now()
    if _ibkr_cache["data"] and _ibkr_cache["time"] and (now - _ibkr_cache["time"]).seconds < 60:
        return _ibkr_cache["data"]
    data = _get_ibkr_status()
    _ibkr_cache["data"] = data
    _ibkr_cache["time"] = now
    return data


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

@app.get("/api/ibkr")
def api_ibkr():
    return _cached_ibkr()

@app.get("/api/botlog")
def api_botlog():
    return {"log": _get_bot_log()}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    sig = _cached_signal()
    bt = _cached_backtest()
    ibkr = _cached_ibkr()
    logs = _get_trade_logs()
    cron_log = _get_cron_log(30)
    bot_log = _get_bot_log(40)
    tz_sthlm = ZoneInfo("Europe/Stockholm")
    now = datetime.now(tz_sthlm)

    action = sig.get("action", "unknown")
    trend = sig.get("trend", "?")
    spy_price = sig.get("spy_price", 0)
    spy_sma = sig.get("spy_sma", 0)
    spy_pct = sig.get("spy_pct_vs_sma", 0)
    holdings = sig.get("target_holdings", [])
    mom_scores = sig.get("momentum_scores", {})
    sig_date = sig.get("date", "?")
    exposure = sig.get("exposure", 1.0)
    realized_vol = sig.get("realized_vol", 0)
    days_in_regime = sig.get("days_in_regime", "?")
    spy_history = sig.get("spy_history", [])

    if action == "hold":
        status_color = "#22c55e"
        status_text = "INVESTED"
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

    tz_et = ZoneInfo("America/New_York")
    now_et = datetime.now(tz_et)
    today_et = now_et.replace(hour=15, minute=45, second=0, microsecond=0)
    if now_et > today_et or now_et.weekday() >= 5:
        next_run_et = today_et + timedelta(days=1)
        while next_run_et.weekday() >= 5:
            next_run_et += timedelta(days=1)
    else:
        next_run_et = today_et
    next_run_sthlm = next_run_et.astimezone(tz_sthlm)
    next_run_str = next_run_sthlm.strftime("%A %b %d, %H:%M CET")
    time_until = next_run_et - now_et
    hours_until = int(time_until.total_seconds() // 3600)
    mins_until = int((time_until.total_seconds() % 3600) // 60)

    # SPY chart data for JS
    chart_dates = json.dumps([d["date"] for d in spy_history])
    chart_prices = json.dumps([d["price"] for d in spy_history])
    chart_sma = json.dumps([d["sma"] for d in spy_history])

    # IBKR status
    ibkr_connected = ibkr.get("connected", False)
    ibkr_equity = ibkr.get("equity", 0)
    ibkr_currency = ibkr.get("currency", "?")
    ibkr_positions = ibkr.get("positions", [])
    ibkr_summary = ibkr.get("summary", {})

    gw_color = "#22c55e" if ibkr_connected else "#ef4444"
    gw_text = "CONNECTED" if ibkr_connected else "OFFLINE"

    cash_val = ibkr_summary.get("TotalCashValue", {}).get("value", 0)
    gross_pos = ibkr_summary.get("GrossPositionValue", {}).get("value", 0)
    unrealized = ibkr_summary.get("UnrealizedPnL", {}).get("value", 0)

    pos_rows = ""
    for p in ibkr_positions:
        pos_rows += f"<tr><td>{p['symbol']}</td><td class='num'>{int(p['qty'])}</td><td class='num'>${p['avg_cost']:.2f}</td></tr>"

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
    for entry in logs[:20]:
        ts = entry.get("timestamp", "?")[:16]
        is_live = entry.get("live", False)
        mode = "LIVE" if is_live else "DRY"
        sig_a = entry.get("signal", {}).get("action", "?")
        orders = entry.get("orders_planned", entry.get("orders", []))
        n_errs = len(entry.get("errors", []))
        order_str = ", ".join(f"{o.get('side','')} {o.get('quantity','')} {o.get('symbol','')}" for o in orders) if orders else "no trades"
        err_badge = f' <span class="err-badge">{n_errs} err</span>' if n_errs else ""
        log_rows += f"<tr><td>{ts}</td><td><span class='tag {mode.lower()}'>{mode}</span></td><td>{sig_a}</td><td>{order_str}{err_badge}</td></tr>"

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
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0a0a0a;color:#e5e5e5;padding:20px;max-width:1200px;margin:0 auto}}
h1{{font-size:1.4rem;font-weight:600;margin-bottom:4px;color:#fff}}
.subtitle{{color:#737373;font-size:.85rem;margin-bottom:20px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}}
.grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px}}
.card{{background:#171717;border:1px solid #262626;border-radius:10px;padding:18px}}
.card h2{{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:#737373;margin-bottom:10px}}
.status-box{{text-align:center;padding:24px 16px}}
.status-dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;animation:pulse 2s infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.4}}}}
.status-label{{font-size:1.6rem;font-weight:700;color:{status_color}}}
.status-detail{{color:#a3a3a3;margin-top:5px;font-size:.9rem}}
.kpi{{text-align:center}}
.kpi .val{{font-size:1.4rem;font-weight:700;color:#fff}}
.kpi .val.sm{{font-size:1.1rem}}
.kpi .label{{font-size:.7rem;color:#737373;margin-top:2px}}
.kpi-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:10px}}
.kpi-grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:10px}}
table{{width:100%;border-collapse:collapse;font-size:.82rem}}
th{{text-align:left;color:#737373;font-weight:500;padding:7px 8px;border-bottom:1px solid #262626;font-size:.72rem;text-transform:uppercase}}
td{{padding:6px 8px;border-bottom:1px solid #1a1a1a}}
tr.held td{{background:#052e16}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.bar{{height:5px;border-radius:3px;min-width:2px}}
.badge{{background:#166534;color:#4ade80;font-size:.62rem;padding:2px 5px;border-radius:4px;margin-left:4px;font-weight:600}}
.err-badge{{background:#7f1d1d;color:#fca5a5;font-size:.62rem;padding:2px 5px;border-radius:4px;margin-left:4px;font-weight:600}}
.tag{{font-size:.68rem;padding:2px 7px;border-radius:4px;font-weight:600}}
.tag.live{{background:#7c2d12;color:#fb923c}}
.tag.dry{{background:#1e3a5f;color:#60a5fa}}
pre{{background:#111;padding:10px;border-radius:6px;font-size:.75rem;overflow-x:auto;max-height:250px;overflow-y:auto;color:#a3a3a3;line-height:1.45}}
.refresh{{color:#525252;font-size:.72rem;margin-top:14px;text-align:center}}
.refresh a{{color:#525252}}
.spy-info{{display:flex;gap:16px;justify-content:center;margin-top:8px;flex-wrap:wrap}}
.spy-info span{{font-size:.82rem}}
.spy-info .lbl{{color:#737373}}
.full{{grid-column:1/-1}}
.gw-status{{display:inline-block;padding:3px 10px;border-radius:4px;font-weight:700;font-size:.8rem}}
canvas{{width:100%!important;max-height:220px}}
.tabs{{display:flex;gap:2px;margin-bottom:10px}}
.tab{{padding:6px 14px;background:#262626;border:none;color:#a3a3a3;cursor:pointer;font-size:.75rem;border-radius:6px 6px 0 0}}
.tab.active{{background:#171717;color:#fff}}
.tab-content{{display:none}}.tab-content.active{{display:block}}
</style></head><body>

<h1>Momentum Rotation Bot</h1>
<p class="subtitle">Live monitoring &middot; auto-refreshes every 2 min &middot; {now.strftime('%b %d %Y, %H:%M')} Stockholm</p>

<!-- Row 1: Signal + Gateway + Next Run -->
<div class="grid3">

<div class="card status-box">
<h2>Signal</h2>
<div><span class="status-dot" style="background:{status_color}"></span><span class="status-label">{status_text}</span></div>
<div class="status-detail">{status_detail}</div>
<div style="color:#525252;font-size:.72rem;margin-top:6px">{sig_date} &middot; {days_in_regime} days in regime</div>
</div>

<div class="card" style="text-align:center">
<h2>IBKR Gateway</h2>
<div class="gw-status" style="background:{'#052e16' if ibkr_connected else '#450a0a'};color:{gw_color}">{gw_text}</div>
<div class="kpi-grid" style="margin-top:14px">
<div class="kpi"><div class="val sm">{ibkr_equity:,.0f}</div><div class="label">Net Liq ({ibkr_currency})</div></div>
<div class="kpi"><div class="val sm">{cash_val:,.0f}</div><div class="label">Cash</div></div>
<div class="kpi"><div class="val sm">{gross_pos:,.0f}</div><div class="label">Positions</div></div>
</div>
{f'<div style="color:#a3a3a3;font-size:.78rem;margin-top:10px">Unrealized P&L: <span style="color:{chr(35)}22c55e" if unrealized >= 0 else "color:{chr(35)}ef4444"">{unrealized:+,.0f}</span></div>' if unrealized else ''}
</div>

<div class="card" style="text-align:center">
<h2>Next Trade</h2>
<div class="kpi"><div class="val sm" style="color:#60a5fa">{next_run_str}</div><div class="label">in {hours_until}h {mins_until}m</div></div>
<div class="kpi-grid" style="margin-top:12px">
<div class="kpi"><div class="val sm">{exposure:.0%}</div><div class="label">Exposure</div></div>
<div class="kpi"><div class="val sm">{realized_vol:.0f}%</div><div class="label">Vol (ann)</div></div>
<div class="kpi"><div class="val sm">{spy_pct:+.1f}%</div><div class="label">SPY vs SMA</div></div>
</div>
</div>

</div>

<!-- Row 2: SPY Chart (full width) -->
<div class="card full" style="margin-bottom:14px">
<h2>SPY vs SMA(100) Trend Filter</h2>
<canvas id="spyChart" height="200"></canvas>
</div>

<!-- Row 3: Backtest stats -->
<div class="card full" style="margin-bottom:14px">
<h2>Backtest Performance ({bt_years} years)</h2>
<div class="kpi-grid4" style="margin-top:4px">
<div class="kpi"><div class="val sm" style="color:#22c55e">{bt_cagr:+.1f}%</div><div class="label">Strategy CAGR</div></div>
<div class="kpi"><div class="val sm">{bt_spy:+.1f}%</div><div class="label">SPY CAGR</div></div>
<div class="kpi"><div class="val sm" style="color:#22c55e">{bt_alpha:+.1f}%</div><div class="label">Alpha</div></div>
<div class="kpi"><div class="val sm">{bt_sharpe:.2f}</div><div class="label">Sharpe</div></div>
</div>
</div>

<!-- Row 4: Momentum + Positions -->
<div class="grid">

<div class="card">
<h2>Momentum Rankings (20-day)</h2>
<table><thead><tr><th>#</th><th>Symbol</th><th>Momentum</th><th class="num">Return</th></tr></thead>
<tbody>{mom_rows if mom_rows else '<tr><td colspan="4" style="color:#525252">SPY below SMA &mdash; no rankings while in cash</td></tr>'}</tbody></table>
</div>

<div class="card">
<h2>Live Positions (IBKR)</h2>
<table><thead><tr><th>Symbol</th><th class="num">Qty</th><th class="num">Avg Cost</th></tr></thead>
<tbody>{pos_rows if pos_rows else '<tr><td colspan="3" style="color:#525252">No open positions</td></tr>'}</tbody></table>
</div>

</div>

<!-- Row 5: Trade Log -->
<div class="card full" style="margin-bottom:14px">
<h2>Trade History</h2>
<table><thead><tr><th>Time</th><th>Mode</th><th>Signal</th><th>Orders</th></tr></thead>
<tbody>{log_rows if log_rows else '<tr><td colspan="4" style="color:#525252">No trades yet.</td></tr>'}</tbody></table>
</div>

<!-- Row 6: Logs -->
<div class="grid">
<div class="card">
<h2>Cron Log (trade.log)</h2>
<pre>{cron_log}</pre>
</div>
<div class="card">
<h2>Bot Log (bot.log)</h2>
<pre>{bot_log}</pre>
</div>
</div>

<div class="refresh">Auto-refreshes every 2 minutes &middot; <a href="/">Refresh now</a> &middot;
API: <a href="/api/signal">/signal</a> <a href="/api/ibkr">/ibkr</a> <a href="/api/logs">/logs</a></div>

<script>
// Minimal canvas chart - no external dependencies
(function() {{
    const dates = {chart_dates};
    const prices = {chart_prices};
    const sma = {chart_sma};
    if (!dates.length) return;

    const canvas = document.getElementById('spyChart');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;
    const pad = {{t:18, r:55, b:30, l:10}};
    const cW = W - pad.l - pad.r, cH = H - pad.t - pad.b;

    const allVals = prices.concat(sma.filter(v => v !== null));
    const minV = Math.min(...allVals) * 0.995;
    const maxV = Math.max(...allVals) * 1.005;

    function x(i) {{ return pad.l + (i / (dates.length - 1)) * cW; }}
    function y(v) {{ return pad.t + (1 - (v - minV) / (maxV - minV)) * cH; }}

    // Fill zones: green where price > sma, red where price < sma
    for (let i = 1; i < dates.length; i++) {{
        if (sma[i] === null || sma[i-1] === null) continue;
        const above = prices[i] >= sma[i];
        ctx.fillStyle = above ? 'rgba(34,197,94,0.07)' : 'rgba(239,68,68,0.07)';
        ctx.beginPath();
        ctx.moveTo(x(i-1), y(prices[i-1]));
        ctx.lineTo(x(i), y(prices[i]));
        ctx.lineTo(x(i), y(sma[i]));
        ctx.lineTo(x(i-1), y(sma[i-1]));
        ctx.closePath();
        ctx.fill();
    }}

    // Grid lines
    ctx.strokeStyle = '#1a1a1a';
    ctx.lineWidth = 1;
    const nGrid = 5;
    for (let i = 0; i <= nGrid; i++) {{
        const val = minV + (maxV - minV) * (i / nGrid);
        const yy = y(val);
        ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(W - pad.r, yy); ctx.stroke();
        ctx.fillStyle = '#525252'; ctx.font = '11px sans-serif'; ctx.textAlign = 'right';
        ctx.fillText('$' + val.toFixed(0), W - pad.r + 40, yy + 4);
    }}

    // SMA line
    ctx.beginPath();
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    let started = false;
    for (let i = 0; i < dates.length; i++) {{
        if (sma[i] === null) continue;
        if (!started) {{ ctx.moveTo(x(i), y(sma[i])); started = true; }}
        else ctx.lineTo(x(i), y(sma[i]));
    }}
    ctx.stroke();
    ctx.setLineDash([]);

    // Price line
    ctx.beginPath();
    ctx.strokeStyle = '#e5e5e5';
    ctx.lineWidth = 2;
    ctx.moveTo(x(0), y(prices[0]));
    for (let i = 1; i < dates.length; i++) ctx.lineTo(x(i), y(prices[i]));
    ctx.stroke();

    // Legend (top-right, no overlap)
    const legX = W - pad.r - 140;
    ctx.font = '12px sans-serif';
    ctx.fillStyle = 'rgba(23,23,23,0.85)';
    ctx.fillRect(legX - 6, 4, 144, 22);
    ctx.lineWidth = 2; ctx.setLineDash([]);
    ctx.strokeStyle = '#e5e5e5'; ctx.beginPath(); ctx.moveTo(legX, 16); ctx.lineTo(legX + 18, 16); ctx.stroke();
    ctx.fillStyle = '#e5e5e5'; ctx.fillText('SPY', legX + 22, 20);
    ctx.strokeStyle = '#f59e0b'; ctx.setLineDash([6,3]); ctx.beginPath(); ctx.moveTo(legX + 62, 16); ctx.lineTo(legX + 80, 16); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = '#f59e0b'; ctx.fillText('SMA(100)', legX + 84, 20);

    // Date labels
    ctx.fillStyle = '#525252'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
    const step = Math.max(1, Math.floor(dates.length / 6));
    for (let i = 0; i < dates.length; i += step) {{
        ctx.fillText(dates[i].slice(5), x(i), H - 8);
    }}
    ctx.fillText(dates[dates.length-1].slice(5), x(dates.length-1), H - 8);

    // Current price dot
    const lastI = dates.length - 1;
    ctx.beginPath();
    ctx.arc(x(lastI), y(prices[lastI]), 4, 0, Math.PI * 2);
    ctx.fillStyle = prices[lastI] >= (sma[lastI] || 0) ? '#22c55e' : '#ef4444';
    ctx.fill();
}})();
</script>

</body></html>"""
    return html


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Dashboard starting at http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
