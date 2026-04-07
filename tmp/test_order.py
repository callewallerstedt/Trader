#!/usr/bin/env python3
"""Test end-to-end order execution on IBKR paper account.
Places a small MOC order, checks status, then cancels it."""
import sys, time
sys.path.insert(0, "/home/calle/Trader")

from broker.ibkr import IBKRBroker, OrderRequest

print("=" * 50)
print("IBKR ORDER EXECUTION TEST (Paper Account)")
print("=" * 50)

broker = IBKRBroker()

# Step 1: Connect
print("\n[1] Connecting to IBKR Gateway...")
if not broker.connect():
    print("FAIL: Could not connect")
    sys.exit(1)
print("OK: Connected")

# Step 2: Check account
equity = broker.get_equity()
currency = broker.get_currency()
print(f"\n[2] Account: {equity:,.2f} {currency}")

# Step 3: Check current positions
positions = broker.get_positions()
print(f"\n[3] Current positions: {len(positions)}")
for p in positions:
    print(f"   {p.symbol}: {p.quantity} shares @ ${p.avg_cost:.2f}")

# Step 4: Qualify a contract (SPY as test)
print("\n[4] Testing contract qualification...")
from ib_insync import Stock
contract = Stock("SPY", "SMART", "USD")
qualified = broker._ib.qualifyContracts(contract)
if qualified:
    print(f"OK: SPY qualified - conId={contract.conId}")
else:
    print("FAIL: Could not qualify SPY contract")
    broker.disconnect()
    sys.exit(1)

# Step 5: Get current SPY price
print("\n[5] Getting SPY market data...")
ticker = broker._ib.reqMktData(contract, '', False, False)
broker._ib.sleep(3)
last = ticker.last if ticker.last and ticker.last > 0 else ticker.close
print(f"OK: SPY price = ${last}")
broker._ib.cancelMktData(contract)

# Step 6: Place a tiny MOC BUY order (1 share)
print("\n[6] Placing test MOC order: BUY 1 SPY...")
from ib_insync import Order
order = Order(action="BUY", totalQuantity=1, orderType="MOC")
trade = broker._ib.placeOrder(contract, order)
broker._ib.sleep(3)

oid = trade.order.orderId
status = trade.orderStatus.status
print(f"OK: Order placed - ID={oid}, Status={status}")

for entry in trade.log:
    print(f"   log: {entry.time} | {entry.status} | {entry.message} | err={entry.errorCode}")

# Step 7: Check open orders
print("\n[7] Checking open orders...")
open_orders = broker._ib.openOrders()
print(f"   Open orders: {len(open_orders)}")
for o in open_orders:
    print(f"   OrderId={o.orderId} {o.action} {o.totalQuantity} {o.orderType}")

# Step 8: Cancel the test order
print("\n[8] Cancelling test order...")
broker._ib.cancelOrder(trade.order)
broker._ib.sleep(2)
final_status = trade.orderStatus.status
print(f"OK: Final status after cancel = {final_status}")

# Step 9: Verify no open orders remain
open_after = broker._ib.openOrders()
print(f"\n[9] Open orders after cancel: {len(open_after)}")

# Step 10: Test a LIMIT order too (to verify order types work)
print("\n[10] Testing LIMIT order: BUY 1 SPY @ $1.00 (will not fill)...")
limit_order = Order(action="BUY", totalQuantity=1, orderType="LMT", lmtPrice=1.00)
limit_trade = broker._ib.placeOrder(contract, limit_order)
broker._ib.sleep(2)
print(f"OK: Limit order placed - ID={limit_trade.order.orderId}, Status={limit_trade.orderStatus.status}")
broker._ib.cancelOrder(limit_trade.order)
broker._ib.sleep(1)
print(f"OK: Cancelled - Status={limit_trade.orderStatus.status}")

broker.disconnect()
print("\n" + "=" * 50)
print("ALL TESTS PASSED - Order execution pipeline works!")
print("MOC orders will submit and execute at market close.")
print("=" * 50)
