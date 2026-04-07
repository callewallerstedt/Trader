#!/bin/bash
sleep 5
curl -s http://localhost:8080/api/signal | python3 -c "
import sys, json
d = json.load(sys.stdin)
h = d.get('spy_history', [])
print(f'Points: {len(h)}')
nulls = sum(1 for x in h if x.get('sma') is None)
print(f'SMA nulls: {nulls}/{len(h)}')
if h:
    print('First:', h[0])
    print('Last:', h[-1])
"
