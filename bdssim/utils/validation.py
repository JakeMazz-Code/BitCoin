"""Validation helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from bdssim.config import Config


def assert_fraction(value: float, name: str) -> None:
    """Ensure value lies within [0, 1]."""

    if not 0 <= value <= 1:
        raise ValueError(f"{name} must lie in [0, 1], received {value}")


def assert_monotonic(sequence: Sequence[float], increasing: bool = True, tol: float = 1e-9) -> None:
    """Ensure a sequence is monotonic within tolerance."""

    arr = np.asarray(sequence, dtype=float)
    diffs = np.diff(arr)
    if increasing and np.any(diffs < -tol):
        raise ValueError("Sequence must be non-decreasing")
    if not increasing and np.any(diffs > tol):
        raise ValueError("Sequence must be non-increasing")


def validate_config(config: Config) -> None:
    """Run unit checks on configuration (volumes, caps, venues)."""

    assert_fraction(config.policy_us.max_frac_adv, "policy_us.max_frac_adv")
    assert_fraction(config.policy_us.venue_mix.cex, "policy_us.venue_mix.cex")
    assert_fraction(config.policy_us.venue_mix.otc, "policy_us.venue_mix.otc")
    if config.market.adv_usd <= 0:
        raise ValueError("ADV must be positive")
    if config.objective.horizon_days <= 0:
        raise ValueError("Horizon must be positive")


__all__ = ["assert_fraction", "assert_monotonic", "validate_config"]
