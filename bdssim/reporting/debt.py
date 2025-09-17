"""Debt projection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


def project_debt(baseline: float, annual_cagr: float, years: float) -> float:
    """Compound baseline debt forward by ``years`` at ``annual_cagr``."""

    if baseline < 0:
        raise ValueError("baseline must be non-negative")
    return float(baseline * (1 + annual_cagr) ** years)


@dataclass(frozen=True)
class DebtProjectionRow:
    years: float
    projected_debt: float


def build_debt_projection(baseline: float, annual_cagr: float, years: Iterable[float]) -> List[DebtProjectionRow]:
    return [DebtProjectionRow(years=yr, projected_debt=project_debt(baseline, annual_cagr, yr)) for yr in years]


__all__ = ["project_debt", "DebtProjectionRow", "build_debt_projection"]
