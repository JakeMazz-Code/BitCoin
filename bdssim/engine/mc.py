"""Monte Carlo driver integrating bdssim simulation and ceiling stats."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from bdssim.engine.simulation import run_years
from bdssim.engine.stats import MCRun, mean, quantiles


@dataclass
class _RunBuffers:
    price_final: List[float]
    float_final: List[float]
    reflexivity_final: List[float]
    price_paths: List[List[float]]
    float_paths: List[List[float]]
    reserve_paths: List[List[float]]


def _config_hash(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _reserve_value(row: pd.Series) -> float:
    price = float(row.get("price", float("nan")))
    tradable = float(row.get("tradable_float", row.get("effective_float", float("nan"))))
    if pd.isna(price) or pd.isna(tradable):
        return float("nan")
    return price * tradable


def run_mc(config: dict, *, draws: int = 256, seed: int = 42) -> MCRun:
    """Run Monte Carlo draws around ``config`` and return aggregated statistics."""

    if draws <= 0:
        raise ValueError("draws must be positive")
    if config is None:
        raise ValueError("config cannot be None")

    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    buffers = _RunBuffers(price_final=[], float_final=[], reflexivity_final=[], price_paths=[], float_paths=[], reserve_paths=[])

    years_index: List[int] | None = None
    for _ in range(draws):
        draw_cfg = dict(config)
        draw_cfg["seed"] = int(rng.integers(0, 2**31 - 1))
        df = run_years(config=draw_cfg)
        if years_index is None:
            years_index = df["year"].astype(int).tolist()
        else:
            if len(df) != len(years_index):
                raise ValueError("Inconsistent horizon across draws")
        buffers.price_final.append(float(df["price"].iloc[-1]))
        effective_col = "tradable_float" if "tradable_float" in df.columns else "effective_float"
        buffers.float_final.append(float(df[effective_col].iloc[-1]))
        buffers.reflexivity_final.append(float(df.get("reflexivity_delta", pd.Series([0.0] * len(df))).iloc[-1]))
        buffers.price_paths.append([float(val) for val in df["price"]])
        buffers.float_paths.append([float(val) for val in df[effective_col]])
        buffers.reserve_paths.append([_reserve_value(row) for _, row in df.iterrows()])

    if years_index is None:
        raise RuntimeError("No draws executed")

    price_percentiles = quantiles(buffers.price_final)
    float_percentiles = quantiles(buffers.float_final)

    series_price: List[float] = []
    series_float: List[float] = []
    series_reserve: List[float] = []
    draws_float_np = np.array(buffers.float_paths)
    draws_price_np = np.array(buffers.price_paths)
    draws_reserve_np = np.array(buffers.reserve_paths)
    for year_idx in range(len(years_index)):
        series_price.append(float(np.mean(draws_price_np[:, year_idx])))
        series_float.append(float(np.mean(draws_float_np[:, year_idx])))
        series_reserve.append(float(np.nanmean(draws_reserve_np[:, year_idx])))

    meta = {
        "seed": seed,
        "draws": draws,
        "runtime_sec": time.perf_counter() - start,
        "config_hash": _config_hash(config),
        "years": years_index,
    }

    percentiles = {
        "price": price_percentiles,
        "tradable_float": float_percentiles,
        "reflexivity_delta": quantiles(buffers.reflexivity_final),
    }

    series = {
        "price": series_price,
        "tradable_float": series_float,
        "reserve_value": series_reserve,
    }

    return MCRun(percentiles=percentiles, series=series, meta=meta)


__all__ = ["run_mc"]
