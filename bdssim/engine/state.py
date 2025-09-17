"""Simulation state definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class MarketState:
    """Container for mutable market state."""

    day: int
    price: float
    sigma: float
    adv_usd: float
    depth_score: float
    liquidity_score: float
    momentum: float = 0.0
    total_order_usd: float = 0.0
    visible_order_usd: float = 0.0
    adopters: int = 0
    _return_history: List[float] = field(default_factory=list)

    def update_price(self, new_price: float) -> None:
        if self.price > 0:
            daily_return = new_price / self.price - 1.0
            self._return_history.append(daily_return)
            self._return_history = self._return_history[-30:]
            if self._return_history:
                self.momentum = float(np.mean(self._return_history))
        self.price = new_price

    def update_sigma(self, sigma: float) -> None:
        self.sigma = sigma

    def update_liquidity(self, adv_usd: float, depth_score: float) -> None:
        self.adv_usd = adv_usd
        self.depth_score = depth_score
        self.liquidity_score = 0.5 * depth_score + 0.5 * min(adv_usd / 20_000_000_000, 1.0)


__all__ = ["MarketState"]
