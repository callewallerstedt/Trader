#!/bin/bash
echo "=== Gateway ==="
pgrep -f "java.*ibcalpha" > /dev/null && echo "RUNNING" || echo "DEAD"

echo ""
echo "=== Dashboard ==="
curl -s http://localhost:8080/api/signal 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'Signal: {d.get(\"action\")} | Trend: {d.get(\"trend\")} | SPY: {d.get(\"spy_price\")} | SMA: {d.get(\"spy_sma\")}')
spy_hist = d.get('spy_history', [])
print(f'SPY history points: {len(spy_hist)}')
if spy_hist:
    print(f'  Last entry: {spy_hist[-1]}')
    sma_nulls = sum(1 for h in spy_hist if h.get(\"sma\") is None)
    print(f'  SMA null entries: {sma_nulls} of {len(spy_hist)}')
" 2>/dev/null || echo "Dashboard not responding"

echo ""
echo "=== IBKR ==="
curl -s http://localhost:8080/api/ibkr 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'Connected: {d.get(\"connected\")} | Equity: {d.get(\"equity\")} {d.get(\"currency\")}')
print(f'Positions: {d.get(\"positions\", [])}')
" 2>/dev/null || echo "IBKR check failed"

echo ""
echo "=== Cron ==="
crontab -l 2>/dev/null | grep -i trade || echo "No cron"
