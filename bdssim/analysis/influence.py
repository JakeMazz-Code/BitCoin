"""Influence maximisation heuristics for bdssim adoption network."""

from __future__ import annotations

from typing import List

import networkx as nx

from bdssim.adoption.countries import CountryCatalog


def greedy_seed_selection(catalog: CountryCatalog, k: int = 3, weight_attr: str = "weight") -> List[str]:
    """Select k seed countries using a weighted degree heuristic."""

    graph = catalog.graph
    scores: dict[str, float] = {}
    for node in catalog.codes():
        score = 0.0
        if isinstance(graph, nx.DiGraph):
            for _, _, data in graph.out_edges(node, data=True):
                score += data.get(weight_attr, 0.0)
            for _, _, data in graph.in_edges(node, data=True):
                score += data.get(weight_attr, 0.0)
        else:
            for neighbor in graph.neighbors(node):
                edge_data = graph.get_edge_data(node, neighbor) or {}
                score += edge_data.get(weight_attr, 0.0)
        scores[node] = score
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [code for code, _ in ranked[:k]]


__all__ = ["greedy_seed_selection"]
