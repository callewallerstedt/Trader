#!/usr/bin/env python3
"""Test qualifying all target stocks for tonight's trade."""
import sys
sys.path.insert(0, "/home/calle/Trader")
from broker.ibkr import IBKRBroker

targets = ["XOM", "XLE", "INTC", "JNJ", "AMD"]

broker = IBKRBroker()
if not broker.connect():
    print("FAIL: Could not connect")
    sys.exit(1)

from ib_insync import Stock

print("Testing contract qualification for tonight's targets:\n")
all_ok = True
for sym in targets:
    contract = Stock(sym, "SMART", "USD")
    qualified = broker._ib.qualifyContracts(contract)
    if qualified:
        print(f"  {sym:6s} OK  conId={contract.conId}  exchange={contract.primaryExchange}")
    else:
        print(f"  {sym:6s} FAIL - cannot qualify!")
        all_ok = False

# Also test that we can get account values needed for position sizing
equity = broker.get_equity()
currency = broker.get_currency()
print(f"\nAccount: {equity:,.2f} {currency}")

# Test getting the summary for all the fields the dashboard needs
summary = broker.get_account_summary()
for tag, info in summary.items():
    print(f"  {tag}: {info['value']:,.2f} {info['currency']}")

broker.disconnect()

if all_ok:
    print("\nALL STOCKS QUALIFIED - Tonight's trades will work!")
else:
    print("\nWARNING: Some stocks failed qualification!")
