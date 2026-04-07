#!/bin/bash
echo "=== Dashboard ==="
curl -s http://localhost:8080/api/signal | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Action: {d.get(\"action\")}')
print(f'Trend: {d.get(\"trend\")} Strength: {d.get(\"trend_strength\", 0):.0f}%')
print(f'SPY: {d.get(\"spy_price\", 0):.2f} SMA: {d.get(\"spy_sma\", 0):.2f}')
print(f'Holdings: {d.get(\"target_holdings\", [])}')
print(f'Strategy: {d.get(\"strategy\", \"?\")}')
h = d.get('spy_history', [])
nulls = sum(1 for x in h if x.get('sma') is None)
print(f'Chart: {len(h)} points, {nulls} SMA nulls')
"

echo ""
echo "=== Backtest ==="
curl -s http://localhost:8080/api/backtest | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'CAGR: {d.get(\"cagr_pct\", 0):+.1f}%')
print(f'Alpha: {d.get(\"alpha_pct\", 0):+.1f}%')
print(f'Sharpe: {d.get(\"sharpe\", 0):.2f}')
print(f'MaxDD: {d.get(\"max_drawdown_pct\", 0):.1f}%')
print(f'Years: {d.get(\"years\", 0):.1f}')
"

echo ""
echo "=== IBKR ==="
curl -s http://localhost:8080/api/ibkr | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Connected: {d.get(\"connected\")}')
print(f'Equity: {d.get(\"equity\", 0):,.2f} {d.get(\"currency\", \"?\")}')
"
