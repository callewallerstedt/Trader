#!/usr/bin/env python3
import sys, traceback
sys.path.insert(0, '/home/calle/Trader')
from broker.ibkr import IBKRBroker
try:
    b = IBKRBroker(client_id=50)
    ok = b.connect()
    print(f"Connected: {ok}")
    if ok:
        eq = b.get_equity()
        cur = b.get_currency()
        print(f"Equity: {eq} {cur}")
        b.disconnect()
except Exception as e:
    traceback.print_exc()
