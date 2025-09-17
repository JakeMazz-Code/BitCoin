"""Adoption gating utilities."""

from __future__ import annotations


def unlocked_for_adoption(t: float, us_progress: float, min_lag_years: float, progress_threshold: float) -> bool:
    """Return True when adoption lag and progress thresholds are satisfied."""

    if min_lag_years > 0 and t < min_lag_years:
        return False
    if progress_threshold <= 0:
        return True
    return us_progress >= progress_threshold


__all__ = ["unlocked_for_adoption"]
