"""Country metadata loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import networkx as nx
import pandas as pd


@dataclass(frozen=True)
class Country:
    """Country level metadata relevant for adoption dynamics."""

    code: str
    name: str
    gdp_usd: float
    reserves_usd: float
    bloc: str
    liquidity_score: float
    base_threshold: float


class CountryCatalog:
    """Maintain lookup tables for countries and adjacency graphs."""

    def __init__(self, countries: Iterable[Country], graph: nx.Graph):
        self.countries: Dict[str, Country] = {c.code: c for c in countries}
        self.graph = graph

    def codes(self) -> List[str]:
        return list(self.countries.keys())

    def country(self, code: str) -> Country:
        return self.countries[code]


def load_countries(path: str | Path) -> List[Country]:
    df = pd.read_csv(path)
    required = {"code", "name", "gdp_usd", "reserves_usd", "bloc", "liquidity_score", "base_threshold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Country CSV missing columns: {sorted(missing)}")
    countries = [
        Country(
            code=row["code"],
            name=row["name"],
            gdp_usd=float(row["gdp_usd"]),
            reserves_usd=float(row["reserves_usd"]),
            bloc=row["bloc"],
            liquidity_score=float(row["liquidity_score"]),
            base_threshold=float(row["base_threshold"]),
        )
        for _, row in df.iterrows()
    ]
    return countries


def load_country_graph(path: str | Path, country_codes: Iterable[str]) -> nx.Graph:
    df = pd.read_csv(path)
    required = {"src", "dst", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Country graph CSV missing columns: {sorted(missing)}")
    graph = nx.Graph()
    for code in country_codes:
        graph.add_node(code)
    for _, row in df.iterrows():
        src = row["src"]
        dst = row["dst"]
        if src not in graph or dst not in graph:
            continue
        weight = float(row["weight"])
        graph.add_edge(src, dst, weight=weight)
    return graph


__all__ = ["Country", "CountryCatalog", "load_countries", "load_country_graph"]
