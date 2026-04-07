#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/calle/Trader")
from broker.ibkr import IBKRBroker
b = IBKRBroker()
ok = b.connect()
if ok:
    eq = b.get_equity()
    cur = b.get_currency()
    pos = b.get_positions()
    b.disconnect()
    print(f"Connected: True | Equity: {eq:,.2f} {cur} | Positions: {len(pos)}")
else:
    print("Connected: False")
