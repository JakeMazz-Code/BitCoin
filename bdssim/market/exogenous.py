"""Exogenous flow models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def _load_flow_map(path: str | Path) -> Dict[int, float]:
    df = pd.read_csv(path)
    if "day" not in df.columns or "usd_flow" not in df.columns:
        raise ValueError("Flow CSV must include 'day' and 'usd_flow' columns")
    return dict(zip(df["day"].astype(int), df["usd_flow"].astype(float)))


@dataclass
class ExogenousFlows:
    """Hold deterministic exogenous flows keyed by day."""

    flows: Dict[int, float]

    def get(self, day: int) -> float:
        return self.flows.get(day, 0.0)


@dataclass
class ExogenousManager:
    """Aggregate multiple flow sources into a single time series."""

    series: Iterable[ExogenousFlows]

    def flow_for_day(self, day: int) -> float:
        return float(sum(series.get(day) for series in self.series))


def build_exogenous_manager(etf_csv: str | None, auction_csv: str | None) -> ExogenousManager:
    series = []
    if etf_csv:
        series.append(ExogenousFlows(_load_flow_map(etf_csv)))
    if auction_csv:
        series.append(ExogenousFlows(_load_flow_map(auction_csv)))
    return ExogenousManager(series)


__all__ = ["ExogenousFlows", "ExogenousManager", "build_exogenous_manager"]
