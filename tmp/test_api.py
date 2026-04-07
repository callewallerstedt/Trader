#!/usr/bin/env python3
from ib_insync import IB
ib = IB()
try:
    ib.connect('127.0.0.1', 4002, clientId=99, timeout=10)
    print('CONNECTED')
    acct = ib.managedAccounts()
    print(f'Account: {acct}')
    ib.disconnect()
except Exception as e:
    print(f'FAIL: {e}')
