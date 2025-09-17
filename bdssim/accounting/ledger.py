"""Accounting ledger for policy trades."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Ledger:
    """Track BTC inventory, cost basis, and P&L."""

    cash_usd: float = 0.0
    holdings_btc: float = 0.0
    fifo_lots: List[List[float]] = field(default_factory=list)
    realized_proceeds_usd: float = 0.0
    realized_cost_usd: float = 0.0

    def buy_usd(self, notional_usd: float, price: float) -> float:
        if price <= 0:
            raise ValueError("Price must be positive")
        btc = notional_usd / price if notional_usd > 0 else 0.0
        if btc <= 0:
            return 0.0
        self.cash_usd -= notional_usd
        self.holdings_btc += btc
        self.fifo_lots.append([btc, price])
        return btc

    def sell_btc(self, btc: float, price: float) -> Tuple[float, float]:
        btc = min(btc, self.holdings_btc)
        if btc <= 0:
            return 0.0, 0.0
        proceeds = btc * price
        cost = 0.0
        remaining = btc
        while remaining > 1e-12 and self.fifo_lots:
            lot = self.fifo_lots[0]
            lot_btc, lot_price = lot
            take = min(remaining, lot_btc)
            cost += take * lot_price
            lot_btc -= take
            remaining -= take
            if lot_btc <= 1e-9:
                self.fifo_lots.pop(0)
            else:
                lot[0] = lot_btc
        self.holdings_btc -= btc
        self.cash_usd += proceeds
        self.realized_proceeds_usd += proceeds
        self.realized_cost_usd += cost
        return proceeds, proceeds - cost

    def current_cost_basis(self) -> float:
        return sum(btc * price for btc, price in self.fifo_lots)

    def average_cost(self) -> float:
        if self.holdings_btc <= 0:
            return 0.0
        return self.current_cost_basis() / self.holdings_btc

    def unrealized_pnl(self, price: float) -> float:
        return self.holdings_btc * price - self.current_cost_basis()

    def debt_coverage(self, debt_baseline_usd: float) -> float:
        if debt_baseline_usd <= 0:
            return 0.0
        net = max(self.realized_proceeds_usd - self.realized_cost_usd, 0.0)
        return net / debt_baseline_usd


__all__ = ["Ledger"]
