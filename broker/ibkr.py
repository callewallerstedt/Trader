"""
Interactive Brokers execution via ib_insync.

Submits MOC (Market-On-Close) orders that fill at the 4 PM closing auction.
Supports paper (port 7497) and live (port 7496) modes.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: int
    order_type: str = "MOC"


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_cost: float


class IBKRBroker:
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self._host = host
        self._port = port
        self._client_id = client_id
        self._ib: Any = None

    def connect(self) -> bool:
        try:
            from ib_insync import IB
            self._ib = IB()
            self._ib.connect(self._host, self._port, clientId=self._client_id)
            log.info(f"Connected to IBKR at {self._host}:{self._port}")
            return True
        except ImportError:
            log.error("ib_insync not installed: pip install ib_insync")
            return False
        except Exception as e:
            log.error(f"IBKR connection failed: {e}")
            return False

    def disconnect(self) -> None:
        if self._ib:
            self._ib.disconnect()
            self._ib = None

    def get_positions(self) -> list[Position]:
        if not self._ib:
            return []
        return [
            Position(p.contract.symbol, p.position, p.avgCost)
            for p in self._ib.positions()
        ]

    def get_equity(self) -> float:
        if not self._ib:
            return 0.0
        for av in self._ib.accountValues():
            if av.tag == "NetLiquidation" and av.currency == "USD":
                return float(av.value)
        return 0.0

    def submit_order(self, req: OrderRequest) -> dict:
        if not self._ib:
            return {"status": "not_connected"}

        from ib_insync import Order, Stock

        contract = Stock(req.symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)

        action = "BUY" if req.side.upper() == "BUY" else "SELL"
        order = Order(action=action, totalQuantity=req.quantity, orderType=req.order_type)
        trade = self._ib.placeOrder(contract, order)
        self._ib.sleep(2)

        return {
            "status": str(trade.orderStatus.status),
            "order_id": trade.order.orderId,
            "symbol": req.symbol,
            "side": req.side,
            "quantity": req.quantity,
            "filled": trade.orderStatus.filled,
            "avg_price": trade.orderStatus.avgFillPrice,
        }
