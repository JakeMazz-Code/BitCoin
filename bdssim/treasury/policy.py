"""Treasury execution policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from bdssim.config import PolicyParams, SellRuleParams, VenueMix
from bdssim.market.execution import TWAPScheduler


@dataclass
class PolicyDecision:
    """Execution decision containing net and visible orders."""

    net_usd: float
    visible_usd: float


@dataclass
class Policy:
    """Base policy definition for sovereign actors."""

    name: str
    params: PolicyParams
    reference_sigma: float
    scheduler: TWAPScheduler = field(init=False)
    executed_bands: List[float] = field(default_factory=list)
    last_sale_day: int = -10_000

    def __post_init__(self) -> None:
        self.scheduler = TWAPScheduler(
            usd_per_day=self.params.usd_per_day,
            max_frac_adv=self.params.max_frac_adv,
            slippage_budget_bps=self.params.slippage_budget_bps,
            start_day=self.params.start_day,
            Y=1.0,  # actual impact Y applied later
            reference_sigma=self.reference_sigma,
            smart_pacing=self.params.smart_pacing,
        )

    @property
    def venue_mix(self) -> VenueMix:
        return self.params.venue_mix

    @property
    def sell_rules(self) -> SellRuleParams:
        return self.params.sell_rules

    def buy_order(self, day: int, price_sigma: float, adv_usd: float, depth_score: float) -> float:
        return self.scheduler.order_for_day(day, adv_usd, price_sigma, depth_score)

    def evaluate_sales(self, day: int, price: float, holdings_btc: float) -> float:
        rules = self.sell_rules
        if not rules.enabled:
            return 0.0
        if holdings_btc <= 0:
            return 0.0
        if not rules.take_profit_bands:
            return 0.0
        net_sale = 0.0
        for band in rules.take_profit_bands:
            if price >= band and band not in self.executed_bands and day - self.last_sale_day >= rules.time_lock_days:
                tranche_btc = holdings_btc * rules.tranche_fraction
                net_sale += tranche_btc * price
                self.executed_bands.append(band)
                self.last_sale_day = day
        return net_sale

    def decision(self, day: int, price: float, sigma: float, adv_usd: float, depth: float, holdings_btc: float) -> PolicyDecision:
        buy = self.buy_order(day, sigma, adv_usd, depth)
        sell = self.evaluate_sales(day, price, holdings_btc)
        net = buy - sell
        visible = net * self.venue_mix.cex
        if net < 0:
            visible = net * self.venue_mix.cex
        return PolicyDecision(net_usd=net, visible_usd=visible)


@dataclass
class USPolicy(Policy):
    """US treasury policy wrapper."""


@dataclass
class CountryPolicy(Policy):
    """Country policy with bespoke start day."""


__all__ = ["PolicyDecision", "Policy", "USPolicy", "CountryPolicy"]
