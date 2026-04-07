#!/bin/bash
curl -s http://localhost:8080/api/ibkr | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Connected: {d.get(\"connected\")}')
print(f'Equity: {d.get(\"equity\", 0)} {d.get(\"currency\", \"?\")}')
print(f'Positions: {d.get(\"positions\", [])}')
"
