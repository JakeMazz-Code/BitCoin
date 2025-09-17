"""Market impact models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List


def square_root_impact(order_usd: float, adv_usd: float, sigma: float, Y: float) -> float:
    if adv_usd <= 0:
        raise ValueError("ADV must be positive")
    if order_usd <= 0 or sigma <= 0:
        return 0.0
    return float(Y * sigma * math.sqrt(order_usd / adv_usd))


def split_impact(relative_impact: float, phi: float) -> tuple[float, float]:
    permanent = relative_impact * phi
    temporary = relative_impact - permanent
    return permanent, temporary


@dataclass
class ImpactPropagator:
    tau_days: float
    lambda_: float = 1.0
    buffer: List[float] = field(default_factory=list)

    def step(self, new_impact: float) -> float:
        if self.tau_days <= 0:
            self.buffer = [new_impact * self.lambda_] if new_impact else []
            return sum(self.buffer)
        decay = math.exp(-1.0 / self.tau_days)
        self.buffer = [value * decay for value in self.buffer if abs(value) > 1e-12]
        if new_impact:
            self.buffer.append(new_impact * self.lambda_)
        return sum(self.buffer)

    def reset(self) -> None:
        self.buffer.clear()


def apply_impact(price: float, relative_impact: float, phi: float) -> tuple[float, float]:
    permanent, temporary = split_impact(relative_impact, phi)
    new_price = price * (1 + permanent)
    return new_price, temporary


def price_update(price: float, permanent_component: float, temporary_component: float, noise: float) -> float:
    return price * (1.0 + permanent_component + temporary_component) * math.exp(noise)


def venue_adjustment(relative_impact: float, venue_mix: Dict[str, float], otc_discount: float) -> float:
    otc = venue_mix.get("otc", 0.0)
    cex = venue_mix.get("cex", 0.0)
    return relative_impact * (otc * otc_discount + cex)


__all__ = [
    "square_root_impact",
    "split_impact",
    "ImpactPropagator",
    "apply_impact",
    "price_update",
    "venue_adjustment",
]
