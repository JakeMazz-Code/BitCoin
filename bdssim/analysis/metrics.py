"""Analytics utilities for bdssim."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd

from bdssim.adoption.countries import CountryCatalog


@dataclass
class CascadeStats:
    first_adopter: str | None
    time_to_half: int | None
    total_adopters: int
    adoption_rate: float


def compute_network_centrality(catalog: CountryCatalog) -> pd.DataFrame:
    """Return centrality metrics (degree, eigenvector, betweenness) for countries."""

    graph = catalog.graph.to_undirected()
    degree = dict(graph.degree())
    try:
        eigen = nx.eigenvector_centrality_numpy(graph, weight="weight")
    except Exception:  # pragma: no cover - fallback when solver fails
        eigen = {code: 0.0 for code in catalog.codes()}
    between = nx.betweenness_centrality(graph, weight="weight")
    rows = []
    for code in catalog.codes():
        rows.append(
            {
                "country": code,
                "degree": degree.get(code, 0.0),
                "eigenvector": float(eigen.get(code, 0.0)),
                "betweenness": float(between.get(code, 0.0)),
                "bloc": catalog.country(code).bloc,
            }
        )
    return pd.DataFrame(rows).sort_values("degree", ascending=False).reset_index(drop=True)


def summarize_cascade(timeseries: pd.DataFrame, adoption_column: str = "adopters") -> CascadeStats:
    """Summarise adoption cascades from timeseries data."""

    if adoption_column not in timeseries.columns:
        return CascadeStats(first_adopter=None, time_to_half=None, total_adopters=0, adoption_rate=0.0)
    adopters_series = timeseries[["day", adoption_column]].dropna()
    total = int(adopters_series[adoption_column].max())
    if total <= 0:
        return CascadeStats(first_adopter=None, time_to_half=None, total_adopters=0, adoption_rate=0.0)
    half_threshold = total / 2
    time_to_half = adopters_series.loc[adopters_series[adoption_column] >= half_threshold, "day"].min()
    adoption_rate = total / max(adopters_series["day"].max(), 1)
    return CascadeStats(first_adopter=None, time_to_half=int(time_to_half) if pd.notna(time_to_half) else None, total_adopters=total, adoption_rate=adoption_rate)


def price_distribution(ts: pd.DataFrame) -> pd.Series:
    """Return distribution summary for price path."""

    price = ts["price"]
    return pd.Series(
        {
            "mean_price": price.mean(),
            "std_price": price.std(),
            "max_price": price.max(),
            "min_price": price.min(),
            "max_drawdown": ((price / price.cummax()) - 1).min(),
        }
    )


def yearly_summary(timeseries: pd.DataFrame, days_per_year: int = 365) -> pd.DataFrame:
    """Aggregate end-of-year metrics for the simulation."""

    if timeseries.empty:
        return pd.DataFrame(columns=["year", "price_end", "coverage_end", "holdings_end", "net_orders_usd"])
    ts = timeseries.copy()
    ts["year"] = (ts["day"] // days_per_year) + 1
    summary = ts.groupby("year").agg(
        price_end=("price", "last"),
        coverage_end=("debt_coverage", "last"),
        holdings_end=("us_holdings_btc", "last"),
        net_orders_usd=("total_order_usd", "sum"),
    )
    summary.reset_index(inplace=True)
    return summary


__all__ = [
    "compute_network_centrality",
    "summarize_cascade",
    "price_distribution",
    "CascadeStats",
    "yearly_summary",
]
