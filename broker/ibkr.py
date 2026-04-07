"""
Interactive Brokers execution via ib_insync.

Submits MOC (Market-On-Close) orders that fill at the 4 PM closing auction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    symbol: str
    side: str
    quantity: int
    order_type: str = "MOC"


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_cost: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0


class IBKRBroker:
    def __init__(self, host: str = "127.0.0.1", port: int = 4002, client_id: int = 1):
        self._host = host
        self._port = port
        self._client_id = client_id
        self._ib: Any = None

    def connect(self) -> bool:
        try:
            import asyncio
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())

            from ib_insync import IB
            self._ib = IB()
            self._ib.connect(self._host, self._port, clientId=self._client_id, timeout=15)
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

    def is_connected(self) -> bool:
        return self._ib is not None and self._ib.isConnected()

    def get_positions(self) -> list[Position]:
        if not self._ib:
            return []
        positions = []
        for p in self._ib.positions():
            pnl = 0.0
            mv = 0.0
            if hasattr(p, "marketValue"):
                mv = float(p.marketValue or 0)
            if hasattr(p, "unrealizedPNL"):
                pnl = float(p.unrealizedPNL or 0)
            positions.append(Position(
                symbol=p.contract.symbol,
                quantity=p.position,
                avg_cost=p.avgCost,
                market_value=mv,
                unrealized_pnl=pnl,
            ))
        return positions

    def get_equity(self) -> float:
        if not self._ib:
            return 0.0
        for av in self._ib.accountValues():
            if av.tag == "NetLiquidation" and av.currency not in ("", "BASE"):
                return float(av.value)
        return 0.0

    def get_currency(self) -> str:
        if not self._ib:
            return "USD"
        for av in self._ib.accountValues():
            if av.tag == "NetLiquidation" and av.currency not in ("", "BASE"):
                return av.currency
        return "USD"

    def get_account_summary(self) -> dict:
        """Get a comprehensive account snapshot."""
        if not self._ib:
            return {}
        result = {}
        tags_of_interest = {
            "NetLiquidation", "TotalCashValue", "GrossPositionValue",
            "AvailableFunds", "BuyingPower", "MaintMarginReq",
            "UnrealizedPnL", "RealizedPnL",
        }
        for av in self._ib.accountValues():
            if av.tag in tags_of_interest and av.currency not in ("", "BASE"):
                result[av.tag] = {"value": float(av.value), "currency": av.currency}
        return result

    def submit_order(self, req: OrderRequest) -> dict:
        if not self._ib:
            return {"status": "not_connected", "symbol": req.symbol}

        from ib_insync import Order, Stock

        try:
            contract = Stock(req.symbol, "SMART", "USD")
            qualified = self._ib.qualifyContracts(contract)
            if not qualified:
                return {"status": "qualify_failed", "symbol": req.symbol}

            action = "BUY" if req.side.upper() == "BUY" else "SELL"
            order = Order(action=action, totalQuantity=req.quantity, orderType=req.order_type)
            trade = self._ib.placeOrder(contract, order)

            self._ib.sleep(3)

            filled = trade.orderStatus.filled
            status = trade.orderStatus.status
            avg_price = trade.orderStatus.avgFillPrice

            log_entries = []
            for entry in trade.log:
                log_entries.append({
                    "time": str(entry.time),
                    "status": entry.status,
                    "message": entry.message,
                    "errorCode": entry.errorCode,
                })

            return {
                "status": status,
                "order_id": trade.order.orderId,
                "symbol": req.symbol,
                "side": req.side,
                "quantity": req.quantity,
                "order_type": req.order_type,
                "filled": filled,
                "remaining": trade.orderStatus.remaining,
                "avg_price": avg_price,
                "log": log_entries,
            }
        except Exception as e:
            log.error(f"Order submission failed for {req.symbol}: {e}")
            return {"status": "exception", "symbol": req.symbol, "error": str(e)}

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        if not self._ib:
            return 0
        open_orders = self._ib.openOrders()
        for order in open_orders:
            self._ib.cancelOrder(order)
        self._ib.sleep(2)
        return len(open_orders)
