#!/usr/bin/env python3
"""
Live monitoring dashboard for the Multi-TF Momentum Rotation Bot.
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
TZ_STHLM = ZoneInfo("Europe/Stockholm")
TZ_ET = ZoneInfo("America/New_York")


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
                {"symbol": p.symbol, "qty": p.quantity, "avg_cost": p.avg_cost,
                 "unrealized_pnl": p.unrealized_pnl}
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

@app.get("/api/refresh")
def api_refresh():
    """Force-refresh all cached data (signal + IBKR)."""
    _signal_cache["data"] = None
    _signal_cache["time"] = None
    _ibkr_cache["data"] = None
    _ibkr_cache["time"] = None
    return {"status": "ok", "message": "Cache cleared, next page load will fetch fresh data"}


def _esc(s: str) -> str:
    """Escape HTML special chars."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


@app.get("/backtest", response_class=HTMLResponse)
def backtest_page():
    bt = _cached_backtest()
    now = datetime.now(TZ_STHLM)

    monthly = bt.get("monthly_returns", {})
    spy_monthly = bt.get("spy_monthly_returns", {})
    yearly = bt.get("yearly_returns", [])
    eq_curve = bt.get("equity_curve", [])

    MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def _heat_color(val: float) -> str:
        if val > 8: return "rgba(16,185,129,0.55)"
        if val > 4: return "rgba(16,185,129,0.35)"
        if val > 1: return "rgba(16,185,129,0.18)"
        if val > 0: return "rgba(16,185,129,0.08)"
        if val > -1: return "rgba(239,68,68,0.08)"
        if val > -4: return "rgba(239,68,68,0.18)"
        if val > -8: return "rgba(239,68,68,0.35)"
        return "rgba(239,68,68,0.55)"

    def _text_color(val: float) -> str:
        if val > 2: return "#34d399"
        if val > 0: return "#6ee7b7"
        if val > -2: return "#fca5a5"
        return "#f87171"

    # Build heatmap rows
    heatmap_rows = ""
    all_years = sorted(monthly.keys())
    for yr in all_years:
        months = monthly[yr]
        yr_total = sum(months.values())
        heatmap_rows += f'<tr><td class="yr-label">{yr}</td>'
        for m in range(1, 13):
            val = months.get(m)
            if val is not None:
                heatmap_rows += (
                    f'<td class="heat-cell" style="background:{_heat_color(val)}">'
                    f'<span style="color:{_text_color(val)}">{val:+.1f}</span></td>'
                )
            else:
                heatmap_rows += '<td class="heat-cell empty"></td>'
        yr_color = _text_color(yr_total)
        yr_bg = _heat_color(yr_total)
        heatmap_rows += f'<td class="heat-cell yr-total" style="background:{yr_bg}"><span style="color:{yr_color};font-weight:700">{yr_total:+.1f}</span></td></tr>'

    # SPY heatmap
    spy_heatmap_rows = ""
    for yr in all_years:
        months = spy_monthly.get(yr, {})
        yr_total = sum(months.values())
        spy_heatmap_rows += f'<tr><td class="yr-label">{yr}</td>'
        for m in range(1, 13):
            val = months.get(m)
            if val is not None:
                spy_heatmap_rows += (
                    f'<td class="heat-cell" style="background:{_heat_color(val)}">'
                    f'<span style="color:{_text_color(val)}">{val:+.1f}</span></td>'
                )
            else:
                spy_heatmap_rows += '<td class="heat-cell empty"></td>'
        yr_color = _text_color(yr_total)
        yr_bg = _heat_color(yr_total)
        spy_heatmap_rows += f'<td class="heat-cell yr-total" style="background:{yr_bg}"><span style="color:{yr_color};font-weight:700">{yr_total:+.1f}</span></td></tr>'

    # Yearly comparison table
    yearly_rows = ""
    for y in yearly:
        s_color = "#34d399" if y["return_pct"] > 0 else "#f87171"
        spy_color = "#34d399" if y["spy_pct"] > 0 else "#f87171"
        a_color = "#34d399" if y["alpha_pct"] > 0 else "#f87171"
        beat = "var(--green)" if y["return_pct"] > y["spy_pct"] else "var(--red)"
        yearly_rows += (
            f'<tr><td class="sym">{y["year"]}</td>'
            f'<td class="num" style="color:{s_color}">{y["return_pct"]:+.1f}%</td>'
            f'<td class="num" style="color:{spy_color}">{y["spy_pct"]:+.1f}%</td>'
            f'<td class="num" style="color:{a_color}">{y["alpha_pct"]:+.1f}%</td>'
            f'<td><span style="color:{beat};font-weight:600">{"Beat" if y["return_pct"] > y["spy_pct"] else "Lost"}</span></td></tr>'
        )

    # Equity curve chart data
    chart_dates = json.dumps([p["date"] for p in eq_curve])
    chart_eq = json.dumps([p["equity"] for p in eq_curve])
    chart_spy = json.dumps([p["spy"] for p in eq_curve])
    chart_dd = json.dumps([p["dd"] for p in eq_curve])

    bt_cagr = bt.get("cagr_pct", 0)
    bt_spy_cagr = bt.get("spy_cagr_pct", 0)
    bt_alpha = bt.get("alpha_pct", 0)
    bt_sharpe = bt.get("sharpe", 0)
    bt_sortino = bt.get("sortino", 0)
    bt_calmar = bt.get("calmar", 0)
    bt_dd = bt.get("max_drawdown_pct", 0)
    bt_years = bt.get("years", 0)
    bt_win = bt.get("monthly_win_rate_pct", 0)
    bt_daily_win = bt.get("daily_win_rate_pct", 0)
    bt_trades = bt.get("total_trades", 0)
    bt_total_ret = bt.get("total_return_pct", 0)
    bt_spy_ret = bt.get("spy_return_pct", 0)
    bt_pf = bt.get("profit_factor", 0)
    bt_vol = bt.get("ann_volatility_pct", 0)
    bt_best = bt.get("best_day_pct", 0)
    bt_worst = bt.get("worst_day_pct", 0)
    bt_neg = bt.get("neg_years", 0)
    bt_final = bt.get("final_equity", 0)
    years_beat = sum(1 for y in yearly if y["return_pct"] > y["spy_pct"])

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Backtest Results | Momentum Bot</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><text y='28' font-size='28'>📊</text></svg>">
<style>
:root {{ --bg:#09090b; --card:#111113; --border:#1e1e22; --text:#e4e4e7; --text-muted:#71717a; --text-dim:#3f3f46; --green:#10b981; --red:#ef4444; --amber:#f59e0b; --blue:#3b82f6; --radius:12px; }}
* {{ margin:0; padding:0; box-sizing:border-box }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Inter','Segoe UI',sans-serif; background:var(--bg); color:var(--text); padding:24px; max-width:1400px; margin:0 auto; -webkit-font-smoothing:antialiased; }}
.header {{ display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:24px; padding-bottom:16px; border-bottom:1px solid var(--border); }}
.header h1 {{ font-size:1.35rem; font-weight:700; color:#fff; letter-spacing:-0.02em; }}
.header .sub {{ color:var(--text-muted); font-size:0.8rem; }}
.header a {{ color:var(--blue); text-decoration:none; font-size:0.82rem; }}
.header a:hover {{ text-decoration:underline; }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:var(--radius); padding:20px; margin-bottom:16px; }}
.card h2 {{ font-size:0.68rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; color:var(--text-muted); margin-bottom:14px; }}
.grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px; }}
.full {{ grid-column:1/-1; }}
.kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; }}
.kpi {{ text-align:center; padding:12px 8px; background:rgba(255,255,255,0.02); border-radius:8px; }}
.kpi .v {{ font-size:1.2rem; font-weight:700; font-variant-numeric:tabular-nums; color:#fff; }}
.kpi .l {{ font-size:0.6rem; color:var(--text-muted); margin-top:3px; text-transform:uppercase; letter-spacing:0.04em; }}
table {{ width:100%; border-collapse:collapse; font-size:0.78rem; }}
th {{ text-align:left; color:var(--text-muted); font-weight:500; padding:8px 6px; border-bottom:1px solid var(--border); font-size:0.65rem; text-transform:uppercase; letter-spacing:0.04em; }}
td {{ padding:6px; border-bottom:1px solid #151518; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; font-family:'SF Mono','Cascadia Code','Consolas',monospace; }}
.sym {{ font-weight:600; color:#fff; }}
.yr-label {{ font-weight:700; color:#fff; white-space:nowrap; padding-right:10px; }}
.heat-cell {{ text-align:center; padding:4px 2px; font-size:0.7rem; font-family:'SF Mono','Consolas',monospace; min-width:52px; border:1px solid #0a0a0b; }}
.heat-cell.empty {{ background:rgba(255,255,255,0.015); }}
.heat-cell span {{ font-size:0.68rem; }}
.yr-total {{ border-left:2px solid var(--border); }}
canvas {{ width:100%!important; }}
.tabs {{ display:flex; gap:2px; }}
.tab {{ padding:8px 18px; background:transparent; border:1px solid var(--border); border-bottom:none; color:var(--text-muted); cursor:pointer; font-size:0.72rem; font-weight:600; border-radius:8px 8px 0 0; transition:all 0.15s; }}
.tab.active {{ background:var(--card); color:#fff; }}
.tab-body {{ background:var(--card); border:1px solid var(--border); border-radius:0 var(--radius) var(--radius) var(--radius); padding:20px; margin-bottom:16px; }}
.tab-content {{ display:none; }}
.tab-content.active {{ display:block; }}
.footer {{ color:var(--text-dim); font-size:0.7rem; text-align:center; margin-top:20px; padding-top:16px; border-top:1px solid var(--border); }}
.footer a {{ color:var(--text-dim); text-decoration:none; }}
@media (max-width:768px) {{ body {{ padding:12px; }} .grid-2 {{ grid-template-columns:1fr; }} .kpi-grid {{ grid-template-columns:repeat(2,1fr); }} }}
</style></head><body>

<div class="header">
  <div>
    <h1>Backtest Results</h1>
    <div class="sub">Multi-TF Momentum (20+60+126d) top-5 &middot; {bt_years} years &middot; 25bps round-trip &middot; $100k initial</div>
  </div>
  <a href="/">&larr; Back to Dashboard</a>
</div>

<!-- KPI Grid -->
<div class="card">
  <h2>Performance Summary</h2>
  <div class="kpi-grid">
    <div class="kpi"><div class="v" style="color:var(--green)">{bt_cagr:+.1f}%</div><div class="l">Strategy CAGR</div></div>
    <div class="kpi"><div class="v">{bt_spy_cagr:+.1f}%</div><div class="l">SPY CAGR</div></div>
    <div class="kpi"><div class="v" style="color:var(--green)">{bt_alpha:+.1f}%</div><div class="l">Alpha</div></div>
    <div class="kpi"><div class="v">{bt_sharpe:.2f}</div><div class="l">Sharpe</div></div>
    <div class="kpi"><div class="v">{bt_sortino:.2f}</div><div class="l">Sortino</div></div>
    <div class="kpi"><div class="v">{bt_calmar:.2f}</div><div class="l">Calmar</div></div>
    <div class="kpi"><div class="v" style="color:var(--red)">{bt_dd:.1f}%</div><div class="l">Max Drawdown</div></div>
    <div class="kpi"><div class="v">{bt_vol:.1f}%</div><div class="l">Ann. Volatility</div></div>
  </div>
  <div class="kpi-grid" style="margin-top:10px">
    <div class="kpi"><div class="v">{bt_total_ret:+,.0f}%</div><div class="l">Total Return</div></div>
    <div class="kpi"><div class="v">{bt_spy_ret:+,.0f}%</div><div class="l">SPY Total Return</div></div>
    <div class="kpi"><div class="v">${bt_final:,.0f}</div><div class="l">Final Equity</div></div>
    <div class="kpi"><div class="v">{bt_trades:,}</div><div class="l">Total Trades</div></div>
    <div class="kpi"><div class="v">{bt_win:.0f}%</div><div class="l">Monthly Win Rate</div></div>
    <div class="kpi"><div class="v">{bt_daily_win:.0f}%</div><div class="l">Daily Win Rate</div></div>
    <div class="kpi"><div class="v">{bt_pf:.2f}</div><div class="l">Profit Factor</div></div>
    <div class="kpi"><div class="v">{bt_neg}</div><div class="l">Negative Years</div></div>
  </div>
  <div class="kpi-grid" style="margin-top:10px">
    <div class="kpi"><div class="v" style="color:var(--green)">{bt_best:+.2f}%</div><div class="l">Best Day</div></div>
    <div class="kpi"><div class="v" style="color:var(--red)">{bt_worst:+.2f}%</div><div class="l">Worst Day</div></div>
    <div class="kpi"><div class="v" style="color:var(--green)">{years_beat}/{len(yearly)}</div><div class="l">Years Beat SPY</div></div>
    <div class="kpi"><div class="v">{bt_trades // max(int(bt_years),1)}</div><div class="l">Trades/Year</div></div>
  </div>
</div>

<!-- Equity Curve -->
<div class="card">
  <h2>Equity Curve &mdash; Strategy vs SPY</h2>
  <canvas id="eqChart" height="260"></canvas>
</div>

<!-- Drawdown Chart -->
<div class="card">
  <h2>Underwater (Drawdown) Chart</h2>
  <canvas id="ddChart" height="160"></canvas>
</div>

<!-- Monthly Returns Heatmap with Tabs -->
<div>
  <div class="tabs">
    <button class="tab active" onclick="switchTab(event,'heatStrat')">Strategy Monthly Returns (%)</button>
    <button class="tab" onclick="switchTab(event,'heatSpy')">SPY Monthly Returns (%)</button>
  </div>
  <div class="tab-body" style="overflow-x:auto">
    <div id="heatStrat" class="tab-content active">
      <table>
        <thead><tr><th>Year</th>{''.join(f'<th style="text-align:center">{m}</th>' for m in MONTHS)}<th style="text-align:center;border-left:2px solid var(--border)">Year</th></tr></thead>
        <tbody>{heatmap_rows}</tbody>
      </table>
    </div>
    <div id="heatSpy" class="tab-content">
      <table>
        <thead><tr><th>Year</th>{''.join(f'<th style="text-align:center">{m}</th>' for m in MONTHS)}<th style="text-align:center;border-left:2px solid var(--border)">Year</th></tr></thead>
        <tbody>{spy_heatmap_rows}</tbody>
      </table>
    </div>
  </div>
</div>

<!-- Yearly Returns Table -->
<div class="card">
  <h2>Annual Returns Comparison</h2>
  <table>
    <thead><tr><th>Year</th><th class="num">Strategy</th><th class="num">SPY</th><th class="num">Alpha</th><th>vs SPY</th></tr></thead>
    <tbody>{yearly_rows}</tbody>
  </table>
</div>

<div class="footer">
  <a href="/">&larr; Back to Dashboard</a> &middot; Backtest uses 25bps round-trip costs, weekly rebalance, SMA(200) trend filter
</div>

<script>
function switchTab(e, id) {{
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  e.target.classList.add('active');
}}

// Equity curve chart
(function() {{
  const dates = {chart_dates};
  const eq = {chart_eq};
  const spy = {chart_spy};
  if (!dates.length) return;

  const canvas = document.getElementById('eqChart');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const pad = {{t:24, r:70, b:32, l:12}};
  const cW = W - pad.l - pad.r, cH = H - pad.t - pad.b;

  const allV = eq.concat(spy);
  const minV = 0;
  const maxV = Math.max(...allV) * 1.05;

  function xp(i) {{ return pad.l + (i / (dates.length - 1)) * cW; }}
  function yp(v) {{ return pad.t + (1 - (v - minV) / (maxV - minV)) * cH; }}

  // Grid
  ctx.strokeStyle = '#1a1a1e'; ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {{
    const val = minV + (maxV - minV) * (i / 5);
    const yy = yp(val);
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(W - pad.r, yy); ctx.stroke();
    ctx.fillStyle = '#3f3f46'; ctx.font = '10px -apple-system,sans-serif'; ctx.textAlign = 'right';
    if (val >= 1e6) ctx.fillText('$' + (val/1e6).toFixed(1) + 'M', W-pad.r+52, yy+4);
    else if (val >= 1e3) ctx.fillText('$' + (val/1e3).toFixed(0) + 'K', W-pad.r+52, yy+4);
    else ctx.fillText('$' + val.toFixed(0), W-pad.r+52, yy+4);
  }}

  // SPY fill
  ctx.beginPath();
  ctx.moveTo(xp(0), yp(spy[0]));
  for (let i = 1; i < dates.length; i++) ctx.lineTo(xp(i), yp(spy[i]));
  ctx.lineTo(xp(dates.length-1), yp(0));
  ctx.lineTo(xp(0), yp(0));
  ctx.closePath();
  ctx.fillStyle = 'rgba(113,113,122,0.06)';
  ctx.fill();

  // Equity fill
  ctx.beginPath();
  ctx.moveTo(xp(0), yp(eq[0]));
  for (let i = 1; i < dates.length; i++) ctx.lineTo(xp(i), yp(eq[i]));
  ctx.lineTo(xp(dates.length-1), yp(0));
  ctx.lineTo(xp(0), yp(0));
  ctx.closePath();
  ctx.fillStyle = 'rgba(16,185,129,0.08)';
  ctx.fill();

  // SPY line
  ctx.beginPath(); ctx.strokeStyle = '#71717a'; ctx.lineWidth = 1.5; ctx.setLineDash([4,3]);
  ctx.moveTo(xp(0), yp(spy[0]));
  for (let i = 1; i < dates.length; i++) ctx.lineTo(xp(i), yp(spy[i]));
  ctx.stroke(); ctx.setLineDash([]);

  // Equity line
  ctx.beginPath(); ctx.strokeStyle = '#10b981'; ctx.lineWidth = 2;
  ctx.moveTo(xp(0), yp(eq[0]));
  for (let i = 1; i < dates.length; i++) ctx.lineTo(xp(i), yp(eq[i]));
  ctx.stroke();

  // Legend
  const lx = pad.l + 10;
  ctx.font = '12px -apple-system,sans-serif';
  ctx.fillStyle = 'rgba(17,17,19,0.92)'; ctx.fillRect(lx-6, 4, 180, 26);
  ctx.strokeStyle = '#1e1e22'; ctx.lineWidth = 1; ctx.strokeRect(lx-6, 4, 180, 26);
  ctx.lineWidth = 2; ctx.strokeStyle = '#10b981'; ctx.beginPath(); ctx.moveTo(lx, 18); ctx.lineTo(lx+20, 18); ctx.stroke();
  ctx.fillStyle = '#10b981'; ctx.fillText('Strategy', lx+24, 22);
  ctx.strokeStyle = '#71717a'; ctx.setLineDash([4,3]); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(lx+90, 18); ctx.lineTo(lx+110, 18); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = '#71717a'; ctx.fillText('SPY', lx+114, 22);

  // Dates
  ctx.fillStyle = '#3f3f46'; ctx.font = '10px -apple-system,sans-serif'; ctx.textAlign = 'center';
  const step = Math.max(1, Math.floor(dates.length / 8));
  for (let i = 0; i < dates.length; i += step) ctx.fillText(dates[i].slice(0,4), xp(i), H-8);
}})();

// Drawdown chart
(function() {{
  const dates = {chart_dates};
  const dd = {chart_dd};
  if (!dates.length) return;

  const canvas = document.getElementById('ddChart');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const pad = {{t:12, r:50, b:28, l:12}};
  const cW = W - pad.l - pad.r, cH = H - pad.t - pad.b;

  const minDD = Math.min(...dd) * 1.1;
  function xp(i) {{ return pad.l + (i / (dates.length - 1)) * cW; }}
  function yp(v) {{ return pad.t + (v / minDD) * cH; }}

  // Grid
  ctx.strokeStyle = '#1a1a1e'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {{
    const val = minDD * (i / 4);
    const yy = yp(val);
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(W-pad.r, yy); ctx.stroke();
    ctx.fillStyle = '#3f3f46'; ctx.font = '10px -apple-system,sans-serif'; ctx.textAlign = 'right';
    ctx.fillText(val.toFixed(0) + '%', W-pad.r+38, yy+4);
  }}

  // Zero line
  ctx.strokeStyle = '#3f3f46'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(W-pad.r, pad.t); ctx.stroke();

  // Fill
  ctx.beginPath();
  ctx.moveTo(xp(0), pad.t);
  for (let i = 0; i < dates.length; i++) ctx.lineTo(xp(i), yp(dd[i]));
  ctx.lineTo(xp(dates.length-1), pad.t);
  ctx.closePath();
  ctx.fillStyle = 'rgba(239,68,68,0.12)';
  ctx.fill();

  // Line
  ctx.beginPath(); ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 1.2;
  ctx.moveTo(xp(0), yp(dd[0]));
  for (let i = 1; i < dates.length; i++) ctx.lineTo(xp(i), yp(dd[i]));
  ctx.stroke();

  // Dates
  ctx.fillStyle = '#3f3f46'; ctx.font = '10px -apple-system,sans-serif'; ctx.textAlign = 'center';
  const step = Math.max(1, Math.floor(dates.length / 8));
  for (let i = 0; i < dates.length; i += step) ctx.fillText(dates[i].slice(0,4), xp(i), H-6);
}})();
</script>
</body></html>"""
    return html


@app.get("/", response_class=HTMLResponse)
def dashboard():
    sig = _cached_signal()
    bt = _cached_backtest()
    ibkr = _cached_ibkr()
    logs = _get_trade_logs()
    cron_log = _get_cron_log(30)
    bot_log = _get_bot_log(40)
    now = datetime.now(TZ_STHLM)

    action = sig.get("action", "unknown")
    trend = sig.get("trend", "?")
    trend_strength = sig.get("trend_strength", 0)
    spy_price = sig.get("spy_price", 0)
    spy_sma = sig.get("spy_sma", 0)
    spy_pct = sig.get("spy_pct_vs_sma", 0)
    holdings = sig.get("target_holdings", [])
    mom_scores = sig.get("momentum_scores", {})
    all_momentum = sig.get("all_momentum", mom_scores)
    all_prices = sig.get("all_prices", {})
    sig_date = sig.get("date", "?")
    exposure = sig.get("exposure", 1.0)
    eff_exposure = sig.get("effective_exposure", exposure)
    realized_vol = sig.get("realized_vol", 0)
    days_in_regime = sig.get("days_in_regime", "?")
    spy_history = sig.get("spy_history", [])

    if action == "hold":
        status_color = "#10b981"
        status_bg = "rgba(16,185,129,0.08)"
        status_text = "INVESTED"
        status_detail = ", ".join(holdings)
    elif action == "go_to_cash":
        status_color = "#f59e0b"
        status_bg = "rgba(245,158,11,0.08)"
        status_text = "CASH"
        status_detail = sig.get("reason", "SPY below SMA")
    elif action == "wait":
        status_color = "#6b7280"
        status_bg = "rgba(107,114,128,0.08)"
        status_text = "WAITING"
        status_detail = sig.get("reason", "")
    else:
        status_color = "#ef4444"
        status_bg = "rgba(239,68,68,0.08)"
        status_text = "ERROR"
        status_detail = sig.get("reason", str(sig))

    now_et = datetime.now(TZ_ET)
    today_et = now_et.replace(hour=15, minute=45, second=0, microsecond=0)
    if now_et > today_et or now_et.weekday() >= 5:
        next_run_et = today_et + timedelta(days=1)
        while next_run_et.weekday() >= 5:
            next_run_et += timedelta(days=1)
    else:
        next_run_et = today_et
    next_run_sthlm = next_run_et.astimezone(TZ_STHLM)
    next_run_str = next_run_sthlm.strftime("%a %b %d, %H:%M")
    time_until = next_run_et - now_et
    total_secs = max(time_until.total_seconds(), 0)
    hours_until = int(total_secs // 3600)
    mins_until = int((total_secs % 3600) // 60)

    chart_dates = json.dumps([d["date"] for d in spy_history])
    chart_prices = json.dumps([d["price"] for d in spy_history])
    chart_sma = json.dumps([d["sma"] for d in spy_history])

    ibkr_connected = ibkr.get("connected", False)
    ibkr_equity = ibkr.get("equity", 0)
    ibkr_currency = ibkr.get("currency", "?")
    ibkr_positions = ibkr.get("positions", [])
    ibkr_summary = ibkr.get("summary", {})

    gw_color = "#10b981" if ibkr_connected else "#ef4444"
    gw_text = "CONNECTED" if ibkr_connected else "OFFLINE"
    gw_bg = "rgba(16,185,129,0.1)" if ibkr_connected else "rgba(239,68,68,0.1)"

    cash_val = ibkr_summary.get("TotalCashValue", {}).get("value", 0)
    gross_pos = ibkr_summary.get("GrossPositionValue", {}).get("value", 0)
    buying_power = ibkr_summary.get("BuyingPower", {}).get("value", 0)
    unrealized = ibkr_summary.get("UnrealizedPnL", {}).get("value", 0)
    realized = ibkr_summary.get("RealizedPnL", {}).get("value", 0)

    # Portfolio positions with allocation % and profit %
    total_pos_value = sum(abs(p.get("qty", 0) * p.get("avg_cost", 0)) for p in ibkr_positions) or 1
    pos_rows = ""
    for p in ibkr_positions:
        pnl = p.get("unrealized_pnl", 0)
        pnl_color = "#10b981" if pnl >= 0 else "#ef4444"
        cost_basis = abs(p["qty"] * p["avg_cost"])
        pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0
        alloc_pct = (cost_basis / ibkr_equity * 100) if ibkr_equity > 0 else 0
        pos_rows += (
            f"<tr><td><span class='sym'>{_esc(p['symbol'])}</span></td>"
            f"<td class='num'>{int(p['qty'])}</td>"
            f"<td class='num'>${p['avg_cost']:.2f}</td>"
            f"<td class='num'>${cost_basis:,.0f}</td>"
            f"<td class='num'>{alloc_pct:.1f}%</td>"
            f"<td class='num' style='color:{pnl_color}'>{pnl:+,.0f}</td>"
            f"<td class='num' style='color:{pnl_color}'>{pnl_pct:+.1f}%</td></tr>"
        )

    # Tonight's trade plan
    current_pos_set = {p["symbol"] for p in ibkr_positions}
    target_set = set(holdings)
    plan_rows = ""
    if action == "hold" and holdings:
        scaled_eq = ibkr_equity * eff_exposure if ibkr_equity > 0 else 100_000 * eff_exposure
        alloc_each = scaled_eq / len(holdings) if holdings else 0

        to_sell = current_pos_set - target_set
        to_buy_new = target_set - current_pos_set
        to_hold = target_set & current_pos_set

        for sym in sorted(to_sell):
            qty = next((int(abs(p["qty"])) for p in ibkr_positions if p["symbol"] == sym), 0)
            plan_rows += (
                f"<tr><td><span class='tag tag-sell'>SELL</span></td>"
                f"<td><span class='sym'>{sym}</span></td>"
                f"<td class='num'>{qty} shares</td>"
                f"<td>Exit position</td></tr>"
            )
        for sym in sorted(to_buy_new):
            price = all_prices.get(sym, sig.get("prices", {}).get(sym, 0))
            qty = int(alloc_each / price) if price > 0 else 0
            plan_rows += (
                f"<tr><td><span class='tag tag-buy'>BUY</span></td>"
                f"<td><span class='sym'>{sym}</span></td>"
                f"<td class='num'>{qty} shares @ ~${price:.2f}</td>"
                f"<td>New position (~${alloc_each:,.0f})</td></tr>"
            )
        for sym in sorted(to_hold):
            price = all_prices.get(sym, sig.get("prices", {}).get(sym, 0))
            desired = int(alloc_each / price) if price > 0 else 0
            current = next((int(abs(p["qty"])) for p in ibkr_positions if p["symbol"] == sym), 0)
            delta = desired - current
            if abs(delta) > 0:
                side = "BUY" if delta > 0 else "SELL"
                tag_cls = "tag-buy" if delta > 0 else "tag-sell"
                plan_rows += (
                    f"<tr><td><span class='tag {tag_cls}'>{side}</span></td>"
                    f"<td><span class='sym'>{sym}</span></td>"
                    f"<td class='num'>{abs(delta)} shares</td>"
                    f"<td>Rebalance ({current} -> {desired})</td></tr>"
                )
            else:
                plan_rows += (
                    f"<tr><td><span class='tag tag-hold'>HOLD</span></td>"
                    f"<td><span class='sym'>{sym}</span></td>"
                    f"<td class='num'>{current} shares</td>"
                    f"<td>No change needed</td></tr>"
                )
    elif action == "go_to_cash" and current_pos_set:
        for sym in sorted(current_pos_set):
            qty = next((int(abs(p["qty"])) for p in ibkr_positions if p["symbol"] == sym), 0)
            plan_rows += (
                f"<tr><td><span class='tag tag-sell'>SELL</span></td>"
                f"<td><span class='sym'>{sym}</span></td>"
                f"<td class='num'>{qty} shares</td>"
                f"<td>Go to cash (trend filter)</td></tr>"
            )

    # ALL momentum rankings
    mom_rows = ""
    max_abs_score = max(abs(s) for s in all_momentum.values()) if all_momentum else 1
    for i, (sym, score) in enumerate(all_momentum.items()):
        is_held = sym in holdings
        is_top5 = i < 5
        badge = ""
        if is_held:
            badge = '<span class="badge-hold">HOLD</span>'
        elif is_top5 and score > 0:
            badge = '<span class="badge-top5">TOP 5</span>'
        bar_w = min(abs(score) / max(max_abs_score, 1) * 100, 100)
        bar_color = "#10b981" if score > 0 else "#ef4444"
        row_cls = "row-held" if is_held else ""
        price = all_prices.get(sym, 0)
        price_str = f"${price:.2f}" if price > 0 else "-"
        filtered = "" if score > 0 else '<span class="badge-neg">NEG</span>'
        mom_rows += (
            f'<tr class="{row_cls}">'
            f'<td class="num" style="color:#525252">{i+1}</td>'
            f'<td>{_esc(sym)} {badge} {filtered}</td>'
            f'<td class="num" style="color:var(--text-dim)">{price_str}</td>'
            f'<td><div class="mom-bar-bg"><div class="mom-bar" '
            f'style="width:{bar_w}%;background:{bar_color}"></div></div></td>'
            f'<td class="num">{score:+.1f}%</td></tr>'
        )

    log_rows = ""
    for entry in logs[:15]:
        ts = entry.get("timestamp", "?")[:16].replace("T", " ")
        is_live = entry.get("live", False)
        mode_cls = "tag-live" if is_live else "tag-dry"
        mode = "LIVE" if is_live else "DRY"
        sig_a = entry.get("signal", {}).get("action", "?")
        orders = entry.get("orders_planned", entry.get("orders", []))
        n_errs = len(entry.get("errors", []))
        order_str = ", ".join(
            f"{o.get('side', '')} {o.get('quantity', '')} {o.get('symbol', '')}"
            for o in orders
        ) if orders else "no trades"
        err_badge = f' <span class="tag-err">{n_errs} err</span>' if n_errs else ""
        log_rows += (
            f"<tr><td style='white-space:nowrap'>{ts}</td>"
            f"<td><span class='tag {mode_cls}'>{mode}</span></td>"
            f"<td>{sig_a}</td>"
            f"<td>{_esc(order_str)}{err_badge}</td></tr>"
        )

    bt_cagr = bt.get("cagr_pct", 0)
    bt_spy = bt.get("spy_cagr_pct", 0)
    bt_alpha = bt.get("alpha_pct", 0)
    bt_sharpe = bt.get("sharpe", 0)
    bt_dd = bt.get("max_drawdown_pct", 0)
    bt_years = bt.get("years", 0)
    bt_win = bt.get("monthly_win_rate_pct", 0)
    bt_trades = bt.get("total_trades", 0)
    bt_total_ret = bt.get("total_return_pct", 0)
    bt_spy_ret = bt.get("spy_return_pct", 0)

    trend_pct = max(0, min(100, trend_strength))
    if trend_pct >= 70:
        trend_color = "#10b981"
    elif trend_pct >= 40:
        trend_color = "#f59e0b"
    else:
        trend_color = "#ef4444"

    unr_color = "#10b981" if unrealized >= 0 else "#ef4444"
    real_color = "#10b981" if realized >= 0 else "#ef4444"

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Momentum Bot</title>
<meta http-equiv="refresh" content="120">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><text y='28' font-size='28'>📈</text></svg>">
<style>
:root {{
  --bg: #09090b;
  --card: #111113;
  --border: #1e1e22;
  --border-hover: #2a2a30;
  --text: #e4e4e7;
  --text-muted: #71717a;
  --text-dim: #3f3f46;
  --green: #10b981;
  --red: #ef4444;
  --amber: #f59e0b;
  --blue: #3b82f6;
  --purple: #8b5cf6;
  --radius: 12px;
}}
* {{ margin:0; padding:0; box-sizing:border-box }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
  background: var(--bg);
  color: var(--text);
  padding: 24px;
  max-width: 1280px;
  margin: 0 auto;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}}

.header {{
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}}
.header h1 {{
  font-size: 1.35rem;
  font-weight: 700;
  color: #fff;
  letter-spacing: -0.02em;
}}
.header .sub {{
  color: var(--text-muted);
  font-size: 0.8rem;
}}
.header .clock {{
  text-align: right;
  font-size: 0.8rem;
  color: var(--text-muted);
}}

.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
.grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
.grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 16px; }}
.full {{ grid-column: 1 / -1; }}

.card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  transition: border-color 0.2s;
}}
.card:hover {{ border-color: var(--border-hover); }}
.card h2 {{
  font-size: 0.68rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
  margin-bottom: 14px;
}}

.signal-card {{
  text-align: center;
  padding: 28px 20px;
  background: {status_bg};
  border-color: {status_color}22;
}}
.signal-dot {{
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: {status_color};
  margin-right: 8px;
  animation: pulse 2s ease-in-out infinite;
}}
@keyframes pulse {{ 0%,100% {{ opacity:1 }} 50% {{ opacity:0.3 }} }}
.signal-label {{
  font-size: 1.8rem;
  font-weight: 800;
  color: {status_color};
  letter-spacing: 0.04em;
}}
.signal-detail {{
  color: var(--text-muted);
  margin-top: 6px;
  font-size: 0.85rem;
}}
.signal-meta {{
  color: var(--text-dim);
  font-size: 0.72rem;
  margin-top: 8px;
}}

.kpi {{ text-align: center; }}
.kpi .v {{
  font-size: 1.3rem;
  font-weight: 700;
  color: #fff;
  font-variant-numeric: tabular-nums;
}}
.kpi .v.sm {{ font-size: 1.05rem; }}
.kpi .l {{
  font-size: 0.65rem;
  color: var(--text-muted);
  margin-top: 3px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}}
.kpi-row {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(90px, 1fr));
  gap: 12px;
  margin-top: 14px;
}}

.gw-pill {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 14px;
  border-radius: 20px;
  font-weight: 700;
  font-size: 0.78rem;
  background: {gw_bg};
  color: {gw_color};
}}
.gw-pill .dot {{
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: {gw_color};
}}

.trend-meter {{
  background: #1a1a1e;
  border-radius: 6px;
  height: 8px;
  overflow: hidden;
  margin-top: 10px;
}}
.trend-fill {{
  height: 100%;
  border-radius: 6px;
  background: {trend_color};
  width: {trend_pct}%;
  transition: width 0.5s ease;
}}
.trend-labels {{
  display: flex;
  justify-content: space-between;
  font-size: 0.65rem;
  color: var(--text-dim);
  margin-top: 3px;
}}

table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8rem;
}}
th {{
  text-align: left;
  color: var(--text-muted);
  font-weight: 500;
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}}
td {{
  padding: 7px 10px;
  border-bottom: 1px solid #151518;
}}
.num {{
  text-align: right;
  font-variant-numeric: tabular-nums;
  font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace;
}}
.sym {{
  font-weight: 600;
  color: #fff;
}}
tr.row-held td {{
  background: rgba(16, 185, 129, 0.06);
}}

.mom-bar-bg {{
  background: #1a1a1e;
  border-radius: 4px;
  height: 6px;
  overflow: hidden;
  width: 100%;
}}
.mom-bar {{
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s;
}}

.badge-hold {{
  background: rgba(16,185,129,0.15);
  color: var(--green);
  font-size: 0.6rem;
  padding: 2px 6px;
  border-radius: 4px;
  margin-left: 5px;
  font-weight: 700;
  letter-spacing: 0.03em;
}}

.tag {{
  font-size: 0.65rem;
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: 700;
  letter-spacing: 0.02em;
}}
.tag-live {{ background: rgba(245,158,11,0.15); color: var(--amber); }}
.tag-dry  {{ background: rgba(59,130,246,0.15); color: var(--blue); }}
.tag-err  {{ background: rgba(239,68,68,0.15); color: var(--red); font-size:0.6rem; padding:2px 6px; border-radius:4px; margin-left:4px; font-weight:700; }}
.tag-buy  {{ background: rgba(16,185,129,0.15); color: var(--green); }}
.tag-sell {{ background: rgba(239,68,68,0.15); color: var(--red); }}
.tag-hold {{ background: rgba(59,130,246,0.12); color: var(--blue); }}
.badge-top5 {{ background: rgba(59,130,246,0.15); color: var(--blue); font-size:0.58rem; padding:2px 5px; border-radius:4px; margin-left:4px; font-weight:700; }}
.badge-neg {{ background: rgba(239,68,68,0.1); color: #ef4444; font-size:0.55rem; padding:1px 4px; border-radius:3px; margin-left:3px; font-weight:600; }}

pre {{
  background: #0c0c0e;
  border: 1px solid var(--border);
  padding: 14px;
  border-radius: 8px;
  font-size: 0.72rem;
  overflow-x: auto;
  max-height: 260px;
  overflow-y: auto;
  color: var(--text-muted);
  line-height: 1.6;
  font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace;
}}

canvas {{ width: 100% !important; max-height: 240px; }}

.bt-row {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}}
.bt-kpi {{
  text-align: center;
  padding: 10px 8px;
  background: rgba(255,255,255,0.02);
  border-radius: 8px;
}}
.bt-kpi .v {{
  font-size: 1.15rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}}
.bt-kpi .l {{
  font-size: 0.62rem;
  color: var(--text-muted);
  margin-top: 2px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}}

.footer {{
  color: var(--text-dim);
  font-size: 0.7rem;
  text-align: center;
  margin-top: 20px;
  padding-top: 16px;
  border-top: 1px solid var(--border);
}}
.footer a {{ color: var(--text-dim); text-decoration: none; }}
.footer a:hover {{ color: var(--text-muted); }}

.refresh-btn {{
  background: rgba(59,130,246,0.12);
  color: var(--blue);
  border: 1px solid rgba(59,130,246,0.25);
  padding: 7px 16px;
  border-radius: 8px;
  font-size: 0.76rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s;
  white-space: nowrap;
}}
.refresh-btn:hover {{ background: rgba(59,130,246,0.2); border-color: rgba(59,130,246,0.4); }}
.refresh-btn.loading {{
  color: var(--text-muted);
  background: rgba(255,255,255,0.04);
  border-color: var(--border);
  pointer-events: none;
}}

.tabs {{ display: flex; gap: 2px; margin-bottom: 0; }}
.tab {{
  padding: 8px 16px;
  background: transparent;
  border: 1px solid var(--border);
  border-bottom: none;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 0.72rem;
  font-weight: 600;
  border-radius: 8px 8px 0 0;
  transition: all 0.15s;
}}
.tab.active {{
  background: var(--card);
  color: #fff;
  border-color: var(--border);
}}
.tab-body {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 0 var(--radius) var(--radius) var(--radius);
  padding: 20px;
}}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}

@media (max-width: 768px) {{
  body {{ padding: 12px; }}
  .grid-2, .grid-3, .grid-4 {{ grid-template-columns: 1fr; }}
  .bt-row {{ grid-template-columns: repeat(2, 1fr); }}
  .header {{ flex-direction: column; gap: 8px; }}
}}
</style></head><body>

<div class="header">
  <div>
    <h1>Multi-TF Momentum Bot</h1>
    <div class="sub">Pre-close rotation &middot; MOC execution &middot; Rebalance weekly</div>
  </div>
  <div style="display:flex;align-items:center;gap:14px">
    <button id="refreshBtn" onclick="doRefresh()" class="refresh-btn">Refresh Data</button>
    <div class="clock">
      {now.strftime('%A, %b %d %Y')}<br>
      <strong>{now.strftime('%H:%M')}</strong> Stockholm
    </div>
  </div>
</div>

<!-- Row 1: Signal + Gateway + Next Trade -->
<div class="grid-3">

<div class="card signal-card">
  <h2>Current Signal</h2>
  <div><span class="signal-dot"></span><span class="signal-label">{status_text}</span></div>
  <div class="signal-detail">{_esc(status_detail)}</div>
  <div class="signal-meta">{sig_date} &middot; {days_in_regime} days in regime &middot; Trend {trend_strength:.0f}%</div>
</div>

<div class="card" style="text-align:center">
  <h2>IBKR Gateway</h2>
  <div class="gw-pill"><span class="dot"></span>{gw_text}</div>
  <div class="kpi-row">
    <div class="kpi"><div class="v sm">{ibkr_equity:,.0f}</div><div class="l">Net Liq ({ibkr_currency})</div></div>
    <div class="kpi"><div class="v sm">{cash_val:,.0f}</div><div class="l">Cash</div></div>
    <div class="kpi"><div class="v sm">{gross_pos:,.0f}</div><div class="l">Invested</div></div>
  </div>
  <div class="kpi-row" style="margin-top:8px">
    <div class="kpi"><div class="v sm" style="color:{unr_color}">{unrealized:+,.0f}</div><div class="l">Unrealized</div></div>
    <div class="kpi"><div class="v sm" style="color:{real_color}">{realized:+,.0f}</div><div class="l">Realized</div></div>
    <div class="kpi"><div class="v sm">{buying_power:,.0f}</div><div class="l">Buying Power</div></div>
  </div>
</div>

<div class="card" style="text-align:center">
  <h2>Next Trade Window</h2>
  <div class="kpi"><div class="v" style="color:var(--blue)">{next_run_str}</div><div class="l">Stockholm time</div></div>
  <div style="color:var(--text-muted);font-size:0.85rem;margin-top:6px">in <strong>{hours_until}h {mins_until}m</strong></div>
  <div class="kpi-row" style="margin-top:14px">
    <div class="kpi"><div class="v sm">{exposure:.0%}</div><div class="l">Vol Exposure</div></div>
    <div class="kpi"><div class="v sm">{eff_exposure:.0%}</div><div class="l">Eff. Exposure</div></div>
    <div class="kpi"><div class="v sm">{realized_vol:.0f}%</div><div class="l">Realized Vol</div></div>
  </div>

  <h2 style="margin-top:18px">Trend Strength</h2>
  <div style="font-size:1.2rem;font-weight:700;color:{trend_color}">{trend_strength:.0f}%</div>
  <div class="trend-meter"><div class="trend-fill"></div></div>
  <div class="trend-labels"><span>Bearish</span><span>Neutral</span><span>Bullish</span></div>
  <div style="color:var(--text-dim);font-size:0.7rem;margin-top:4px">SPY ${spy_price:.2f} &middot; SMA(200) ${spy_sma:.2f} &middot; {spy_pct:+.1f}%</div>
</div>

</div>

<!-- SPY Chart -->
<div class="card full" style="margin-bottom:16px">
  <h2>SPY vs SMA(200) Trend Filter</h2>
  <canvas id="spyChart" height="220"></canvas>
</div>

<!-- Backtest Performance -->
<div class="card full" style="margin-bottom:16px">
  <h2 style="display:flex;justify-content:space-between;align-items:center">
    <span>Backtest Performance &mdash; {bt_years} years, 25bps round-trip</span>
    <a href="/backtest" style="font-size:0.72rem;color:var(--blue);text-decoration:none;text-transform:none;letter-spacing:0;font-weight:500">View Full Backtest &rarr;</a>
  </h2>
  <div class="bt-row">
    <div class="bt-kpi"><div class="v" style="color:var(--green)">{bt_cagr:+.1f}%</div><div class="l">Strategy CAGR</div></div>
    <div class="bt-kpi"><div class="v">{bt_spy:+.1f}%</div><div class="l">SPY CAGR</div></div>
    <div class="bt-kpi"><div class="v" style="color:var(--green)">{bt_alpha:+.1f}%</div><div class="l">Alpha</div></div>
    <div class="bt-kpi"><div class="v">{bt_sharpe:.2f}</div><div class="l">Sharpe</div></div>
  </div>
  <div class="bt-row" style="margin-top:10px">
    <div class="bt-kpi"><div class="v" style="color:var(--red)">{bt_dd:.1f}%</div><div class="l">Max Drawdown</div></div>
    <div class="bt-kpi"><div class="v">{bt_win:.0f}%</div><div class="l">Monthly Win Rate</div></div>
    <div class="bt-kpi"><div class="v">{bt_total_ret:+,.0f}%</div><div class="l">Total Return</div></div>
    <div class="bt-kpi"><div class="v">{bt_trades:,}</div><div class="l">Total Trades</div></div>
  </div>
</div>

<!-- Tonight's Trade Plan -->
<div class="card full" style="margin-bottom:16px">
  <h2>Tonight's Trade Plan &mdash; {next_run_str} Stockholm &mdash; if markets stay as-is</h2>
  <div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:12px">
    <div style="font-size:0.8rem;color:var(--text-muted)">Target: <strong style="color:#fff">{', '.join(holdings) if holdings else 'CASH'}</strong></div>
    <div style="font-size:0.8rem;color:var(--text-muted)">Exposure: <strong style="color:#fff">{eff_exposure:.0%}</strong></div>
    <div style="font-size:0.8rem;color:var(--text-muted)">Account: <strong style="color:#fff">{ibkr_equity:,.0f} {ibkr_currency}</strong></div>
    <div style="font-size:0.8rem;color:var(--text-muted)">Currently holding: <strong style="color:#fff">{', '.join(sorted(current_pos_set)) if current_pos_set else 'nothing'}</strong></div>
  </div>
  <table>
    <thead><tr><th style="width:80px">Action</th><th>Symbol</th><th class="num">Details</th><th>Reason</th></tr></thead>
    <tbody>{plan_rows if plan_rows else '<tr><td colspan="4" style="color:var(--text-dim)">No trades planned &mdash; portfolio matches target or waiting for signal</td></tr>'}</tbody>
  </table>
  <div style="color:var(--text-dim);font-size:0.68rem;margin-top:10px">Orders will be MOC (Market-On-Close), filling at the 4:00 PM ET closing auction. Signal recalculated at execution time with latest prices.</div>
</div>

<!-- Positions + Portfolio side by side -->
<div class="grid-2">

<div class="card">
  <h2>Live Portfolio (IBKR)</h2>
  <table>
    <thead><tr><th>Symbol</th><th class="num">Qty</th><th class="num">Avg Cost</th><th class="num">Value</th><th class="num">Alloc</th><th class="num">P&amp;L</th><th class="num">P&amp;L %</th></tr></thead>
    <tbody>{pos_rows if pos_rows else '<tr><td colspan="7" style="color:var(--text-dim)">No open positions &mdash; fully in cash</td></tr>'}</tbody>
  </table>
  {'<div style="margin-top:12px;padding-top:10px;border-top:1px solid var(--border)">' + '<div style="display:flex;justify-content:space-between;font-size:0.78rem"><span style="color:var(--text-muted)">Cash allocation</span><span style="color:#fff">' + f"{((ibkr_equity - gross_pos) / ibkr_equity * 100) if ibkr_equity > 0 else 100:.0f}% ({ibkr_equity - gross_pos:,.0f} {ibkr_currency})" + '</span></div>' + '<div style="display:flex;justify-content:space-between;font-size:0.78rem;margin-top:4px"><span style="color:var(--text-muted)">Invested</span><span style="color:#fff">' + f"{(gross_pos / ibkr_equity * 100) if ibkr_equity > 0 else 0:.0f}% ({gross_pos:,.0f} {ibkr_currency})" + '</span></div></div>' if ibkr_connected else ''}
</div>

<div class="card">
  <h2>Full Momentum Rankings &mdash; All {len(all_momentum)} Stocks</h2>
  <div style="max-height:420px;overflow-y:auto">
  <table>
    <thead style="position:sticky;top:0;background:var(--card)"><tr><th>#</th><th>Symbol</th><th class="num">Price</th><th>Momentum</th><th class="num">Score</th></tr></thead>
    <tbody>{mom_rows if mom_rows else '<tr><td colspan="5" style="color:var(--text-dim)">Loading momentum data...</td></tr>'}</tbody>
  </table>
  </div>
  <div style="color:var(--text-dim);font-size:0.65rem;margin-top:8px">Green = positive momentum (eligible). NEG = negative momentum (filtered out). Top 5 with positive momentum are selected.</div>
</div>

</div>

<!-- Trade Log -->
<div class="card full" style="margin-bottom:16px">
  <h2>Trade History (last 15 runs)</h2>
  <table>
    <thead><tr><th>Timestamp</th><th>Mode</th><th>Signal</th><th>Orders</th></tr></thead>
    <tbody>{log_rows if log_rows else '<tr><td colspan="4" style="color:var(--text-dim)">No trade logs yet</td></tr>'}</tbody>
  </table>
</div>

<!-- Logs with Tabs -->
<div style="margin-bottom:16px">
  <div class="tabs">
    <button class="tab active" onclick="switchTab(event,'logCron')">Trade Log</button>
    <button class="tab" onclick="switchTab(event,'logBot')">Bot Log</button>
  </div>
  <div class="tab-body">
    <div id="logCron" class="tab-content active"><pre>{_esc(cron_log)}</pre></div>
    <div id="logBot" class="tab-content"><pre>{_esc(bot_log)}</pre></div>
  </div>
</div>

<div class="footer">
  Auto-refreshes every 2 minutes &middot; <a href="/">Refresh now</a> &middot;
  API: <a href="/api/signal">/signal</a> &middot; <a href="/api/ibkr">/ibkr</a> &middot; <a href="/api/backtest">/backtest</a> &middot; <a href="/api/logs">/logs</a>
</div>

<script>
function doRefresh() {{
  const btn = document.getElementById('refreshBtn');
  btn.textContent = 'Fetching...';
  btn.classList.add('loading');
  fetch('/api/refresh')
    .then(r => r.json())
    .then(() => {{ window.location.reload(); }})
    .catch(() => {{ btn.textContent = 'Error'; setTimeout(() => window.location.reload(), 2000); }});
}}

function switchTab(e, id) {{
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  e.target.classList.add('active');
}}

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
    const pad = {{t: 24, r: 58, b: 32, l: 12}};
    const cW = W - pad.l - pad.r, cH = H - pad.t - pad.b;

    const allVals = prices.concat(sma.filter(v => v !== null));
    const minV = Math.min(...allVals) * 0.995;
    const maxV = Math.max(...allVals) * 1.005;

    function xp(i) {{ return pad.l + (i / (dates.length - 1)) * cW; }}
    function yp(v) {{ return pad.t + (1 - (v - minV) / (maxV - minV)) * cH; }}

    // Fill green/red zones between SPY and SMA
    for (let i = 1; i < dates.length; i++) {{
        if (sma[i] === null || sma[i-1] === null) continue;
        const above = prices[i] >= sma[i];
        ctx.fillStyle = above ? 'rgba(16,185,129,0.06)' : 'rgba(239,68,68,0.06)';
        ctx.beginPath();
        ctx.moveTo(xp(i-1), yp(prices[i-1]));
        ctx.lineTo(xp(i), yp(prices[i]));
        ctx.lineTo(xp(i), yp(sma[i]));
        ctx.lineTo(xp(i-1), yp(sma[i-1]));
        ctx.closePath();
        ctx.fill();
    }}

    // Horizontal grid
    ctx.strokeStyle = '#1a1a1e';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {{
        const val = minV + (maxV - minV) * (i / 5);
        const yy = yp(val);
        ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(W - pad.r, yy); ctx.stroke();
        ctx.fillStyle = '#3f3f46'; ctx.font = '11px -apple-system, sans-serif'; ctx.textAlign = 'right';
        ctx.fillText('$' + val.toFixed(0), W - pad.r + 42, yy + 4);
    }}

    // SMA line (dashed, amber)
    ctx.beginPath();
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 1.8;
    ctx.setLineDash([6, 4]);
    let started = false;
    for (let i = 0; i < dates.length; i++) {{
        if (sma[i] === null) continue;
        if (!started) {{ ctx.moveTo(xp(i), yp(sma[i])); started = true; }}
        else ctx.lineTo(xp(i), yp(sma[i]));
    }}
    ctx.stroke();
    ctx.setLineDash([]);

    // SPY price line
    ctx.beginPath();
    ctx.strokeStyle = '#e4e4e7';
    ctx.lineWidth = 2;
    ctx.moveTo(xp(0), yp(prices[0]));
    for (let i = 1; i < dates.length; i++) ctx.lineTo(xp(i), yp(prices[i]));
    ctx.stroke();

    // Legend (top-right)
    const lx = W - pad.r - 160;
    ctx.font = '12px -apple-system, sans-serif';
    ctx.fillStyle = 'rgba(17,17,19,0.92)';
    ctx.fillRect(lx - 8, 4, 168, 26);
    ctx.strokeStyle = 'rgba(30,30,34,1)'; ctx.lineWidth = 1;
    ctx.strokeRect(lx - 8, 4, 168, 26);

    ctx.lineWidth = 2; ctx.setLineDash([]);
    ctx.strokeStyle = '#e4e4e7'; ctx.beginPath(); ctx.moveTo(lx, 18); ctx.lineTo(lx + 20, 18); ctx.stroke();
    ctx.fillStyle = '#e4e4e7'; ctx.fillText('SPY', lx + 24, 22);

    ctx.strokeStyle = '#f59e0b'; ctx.setLineDash([6,4]); ctx.lineWidth = 1.8;
    ctx.beginPath(); ctx.moveTo(lx + 70, 18); ctx.lineTo(lx + 90, 18); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = '#f59e0b'; ctx.fillText('SMA(200)', lx + 94, 22);

    // Date labels
    ctx.fillStyle = '#3f3f46'; ctx.font = '10px -apple-system, sans-serif'; ctx.textAlign = 'center';
    const step = Math.max(1, Math.floor(dates.length / 7));
    for (let i = 0; i < dates.length; i += step) {{
        ctx.fillText(dates[i].slice(5), xp(i), H - 8);
    }}
    ctx.fillText(dates[dates.length-1].slice(5), xp(dates.length-1), H - 8);

    // Current price dot with glow
    const lastI = dates.length - 1;
    const dotColor = prices[lastI] >= (sma[lastI] || 0) ? '#10b981' : '#ef4444';
    ctx.shadowColor = dotColor;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.arc(xp(lastI), yp(prices[lastI]), 4, 0, Math.PI * 2);
    ctx.fillStyle = dotColor;
    ctx.fill();
    ctx.shadowBlur = 0;
}})();
</script>

</body></html>"""
    return html


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Dashboard starting at http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
