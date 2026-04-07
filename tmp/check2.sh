#!/usr/bin/env bash
curl -s http://localhost:8080/api/signal | python3 -c "
import sys, json
d = json.load(sys.stdin)
h = d['spy_history']
n = sum(1 for x in h if x['sma'] is None)
print(f'Points: {len(h)}  SMA nulls: {n}')
print('First:', h[0])
print('Last:', h[-1])
"
