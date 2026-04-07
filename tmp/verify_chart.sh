#!/bin/bash
curl -s http://localhost:8080/api/signal | python3 << 'PYEOF'
import sys, json
d = json.load(sys.stdin)
h = d.get("spy_history", [])
n = sum(1 for x in h if x.get("sma") is None)
print(f"Points: {len(h)}  SMA_nulls: {n}")
if h:
    print(f"First: {h[0]}")
    print(f"Last: {h[-1]}")
PYEOF
