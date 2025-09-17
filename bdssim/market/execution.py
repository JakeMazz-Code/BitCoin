"""Execution scheduling utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

from bdssim.market.impact import square_root_impact


@dataclass
class TWAPScheduler:
    """Simple TWAP scheduler respecting ADV caps and slippage budgets."""

    usd_per_day: float
    max_frac_adv: float
    slippage_budget_bps: float
    start_day: int
    Y: float
    reference_sigma: float
    smart_pacing: bool = True
    cooldown_days: int = 2
    paused_until: int = 0

    def order_for_day(self, day: int, adv_usd: float, sigma: float, depth_score: float) -> float:
        """Return USD notional to execute on ``day``."""

        if day < self.start_day or day < self.paused_until:
            return 0.0
        adv_cap = self.max_frac_adv * adv_usd
        if adv_cap <= 0:
            return 0.0
        target = min(self.usd_per_day, adv_cap)
        order = target
        if self.smart_pacing:
            depth_adjust = 0.5 + 0.7 * max(0.0, min(1.0, depth_score))
            order *= depth_adjust
            if sigma > 0:
                vol_ratio = self.reference_sigma / sigma
                order *= max(0.4, min(1.6, vol_ratio))
        order = min(order, adv_cap)
        est_bps = 10_000 * square_root_impact(order, max(adv_usd, 1.0), sigma, self.Y)
        if est_bps > self.slippage_budget_bps:
            self.paused_until = day + self.cooldown_days
            return 0.0
        return order

    def pause(self, until_day: int) -> None:
        """Pause execution until ``until_day``."""

        self.paused_until = max(self.paused_until, until_day)


__all__ = ["TWAPScheduler"]
