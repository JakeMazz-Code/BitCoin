"""Monte Carlo statistics utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import math


def quantiles(values: Sequence[float], qs: Iterable[float] = (0.1, 0.5, 0.9)) -> Dict[str, float]:
    """Return empirical quantiles keyed by ``q`` (e.g., ``{"p10": value}``)."""

    samples = sorted(float(v) for v in values)
    if not samples:
        raise ValueError("values cannot be empty")
    n = len(samples)
    out: Dict[str, float] = {}
    for q in qs:
        if not 0.0 <= q <= 1.0:
            raise ValueError("quantile probabilities must be in [0,1]")
        idx = q * (n - 1)
        lower = int(math.floor(idx))
        upper = int(math.ceil(idx))
        if lower == upper:
            val = samples[lower]
        else:
            weight = idx - lower
            val = samples[lower] * (1 - weight) + samples[upper] * weight
        out[f"p{int(round(q * 100))}"] = val
    return out


def drawdown(series: Sequence[float]) -> float:
    """Return max drawdown (fraction)."""

    peak = -math.inf
    max_dd = 0.0
    for value in series:
        price = float(value)
        peak = max(peak, price)
        if peak > 0:
            dd = (price - peak) / peak
            max_dd = min(max_dd, dd)
    return max_dd


def mean(values: Sequence[float]) -> float:
    total = 0.0
    count = 0
    for v in values:
        total += float(v)
        count += 1
    if count == 0:
        raise ValueError("values cannot be empty")
    return total / count


def coverage_ratio(simulated: Sequence[float], analytic: float) -> float:
    """Return relative error between simulated mean and analytic expectation."""

    sim_mean = mean(simulated)
    denom = analytic if analytic != 0 else 1e-12
    return abs(sim_mean - analytic) / abs(denom)


@dataclass
class MCRun:
    percentiles: Dict[str, Dict[str, float]]
    series: Dict[str, List[float]]
    meta: Dict[str, Any]


__all__ = ["MCRun", "quantiles", "drawdown", "mean", "coverage_ratio"]
