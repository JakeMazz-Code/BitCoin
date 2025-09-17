"""Threshold network adoption dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import networkx as nx
import numpy as np
from numpy.random import Generator

from bdssim.adoption.countries import CountryCatalog
from bdssim.config import AdoptionThresholdParams


@dataclass
class ThresholdAdoptionModel:
    """Granovetter-style threshold model with policy frictions."""

    catalog: CountryCatalog
    params: AdoptionThresholdParams
    rng: Generator | None = None
    adopters: List[str] = field(default_factory=list)
    thresholds: Dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        codes = self.catalog.codes()
        base = []
        for code in codes:
            country = self.catalog.country(code)
            noise = 0.0
            if self.rng is not None and self.params.theta_std > 0:
                noise = float(self.rng.normal(0, self.params.theta_std))
            base_theta = (country.base_threshold + self.params.default_theta) / 2 + noise
            base.append((code, float(np.clip(base_theta, 0.0, 1.0))))
        self.thresholds = dict(base)

    def peer_share(self, code: str) -> float:
        graph: nx.Graph = self.catalog.graph
        neighbors = graph[code]
        if not neighbors:
            return 0.0
        adopted_weight = 0.0
        total_weight = 0.0
        for neighbor, data in neighbors.items():
            weight = float(data.get("weight", 0.0))
            total_weight += weight
            if neighbor in self.adopters:
                adopted_weight += weight
        if total_weight == 0:
            return 0.0
        return adopted_weight / total_weight

    def step(self, price_momentum: float, liquidity_score: float, allowed: set[str] | None = None) -> List[str]:
        """Compute new adopters using effective thresholds."""

        allowed_set = allowed if allowed is None else set(allowed)
        new_adopters: List[str] = []
        total = len(self.catalog.countries)
        current_share = len(self.adopters) / total if total else 0.0
        penalty_relief = self.params.bloc_relief.reduction(current_share)
        penalty = max(0.0, self.params.policy_penalty - penalty_relief)
        for code in self.catalog.codes():
            if allowed_set is not None and code not in allowed_set:
                continue
            if code in self.adopters or code in new_adopters:
                continue
            country = self.catalog.country(code)
            theta_base = self.thresholds[code]
            peer = self.peer_share(code)
            theta_eff = theta_base
            theta_eff -= self.params.alpha_peer * peer
            theta_eff -= self.params.alpha_momentum * price_momentum
            theta_eff -= self.params.alpha_liquidity * liquidity_score * country.liquidity_score
            theta_eff += penalty
            theta_eff = float(np.clip(theta_eff, 0.0, 1.0))
            if peer >= theta_eff:
                new_adopters.append(code)
        self.adopters.extend(new_adopters)
        return new_adopters


__all__ = ["ThresholdAdoptionModel"]
