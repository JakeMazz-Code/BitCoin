"""Monte Carlo runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np
import pandas as pd

from bdssim.config import Config
from bdssim.engine.simulation import SimulationEngine
from bdssim.utils.io import save_table, write_config_snapshot


@dataclass
class MonteCarloResult:
    metrics: pd.DataFrame
    percentiles: pd.DataFrame
    timepoint_percentiles: pd.DataFrame


_TARGET_DAYS = {2: 2 * 365 - 1, 5: 5 * 365 - 1, 7: 7 * 365 - 1}


def _max_drawdown(series: pd.Series) -> float:
    running_max = series.cummax()
    drawdowns = (series - running_max) / running_max
    return float(drawdowns.min())


def _sample_timepoint(timeseries: pd.DataFrame, target_day: int) -> pd.Series:
    if timeseries.empty:
        raise ValueError("timeseries cannot be empty")
    capped_day = min(target_day, int(timeseries["day"].iloc[-1]))
    subset = timeseries[timeseries["day"] >= capped_day]
    if subset.empty:
        subset = timeseries.iloc[[-1]]
    return subset.iloc[0]


def run_mc(
    config: Config,
    runs: int,
    seeds: Sequence[int] | None = None,
    data_root: str | Path | None = None,
    out_dir: str | Path | None = None,
    playbook: Iterable[dict[str, Any]] | None = None,
) -> MonteCarloResult:
    """Run ``runs`` Monte Carlo simulations and aggregate metrics."""

    if runs <= 0:
        raise ValueError("runs must be positive")
    seeds = list(seeds) if seeds is not None else [config.objective.seed + i for i in range(runs)]
    if len(seeds) < runs:
        raise ValueError("Insufficient seeds provided")
    metrics: List[dict[str, float]] = []
    timepoint_records: List[dict[str, float]] = []
    timeseries_concat: List[pd.DataFrame] = []
    playbook_list = list(playbook) if playbook is not None else None
    for idx in range(runs):
        obj = config.objective.model_copy(update={"seed": seeds[idx]})
        cfg_i = config.model_copy(update={"objective": obj})
        engine = SimulationEngine(cfg_i, data_root=data_root, playbook=playbook_list)
        result = engine.run()
        ts = result.timeseries.copy()
        ts["run"] = idx
        timeseries_concat.append(ts)
        metrics.append(
            {
                "run": idx,
                "final_coverage": ts["debt_coverage"].iloc[-1],
                "final_price": ts["price"].iloc[-1],
                "peak_price": ts["price"].max(),
                "max_drawdown": _max_drawdown(ts["price"]),
                "adopters_final": ts["adopters"].iloc[-1],
                "us_holdings_btc": ts["us_holdings_btc"].iloc[-1],
            }
        )
        for years, target_day in _TARGET_DAYS.items():
            sample = _sample_timepoint(ts, target_day)
            timepoint_records.append(
                {
                    "run": idx,
                    "years": years,
                    "day": float(sample["day"]),
                    "price": float(sample["price"]),
                    "debt_coverage": float(sample["debt_coverage"]),
                    "us_holdings_btc": float(sample["us_holdings_btc"]),
                }
            )
    metrics_df = pd.DataFrame(metrics)
    percentiles = metrics_df.quantile([0.05, 0.5, 0.95]).reset_index().rename(columns={"index": "percentile"})

    timepoint_df = pd.DataFrame(timepoint_records)
    timepoint_percentiles: pd.DataFrame
    if timepoint_df.empty:
        timepoint_percentiles = pd.DataFrame(columns=["years", "percentile", "day", "price", "debt_coverage", "us_holdings_btc"])
    else:
        rows: List[dict[str, float]] = []
        for years in sorted(timepoint_df["years"].unique()):
            grp = timepoint_df[timepoint_df["years"] == years].drop(columns=["run"])
            quantiles = grp.quantile([0.05, 0.5, 0.95])
            for pct, qrow in quantiles.iterrows():
                rows.append(
                    {
                        "years": years,
                        "percentile": pct,
                        "day": float(qrow["day"]),
                        "price": float(qrow["price"]),
                        "debt_coverage": float(qrow["debt_coverage"]),
                        "us_holdings_btc": float(qrow["us_holdings_btc"]),
                    }
                )
        timepoint_percentiles = pd.DataFrame(rows)

    if out_dir is not None:
        out = Path(out_dir)
        save_table(metrics_df, out, "mc_metrics")
        save_table(percentiles, out, "mc_percentiles")
        save_table(pd.concat(timeseries_concat, ignore_index=True), out, "mc_timeseries")
        if not timepoint_percentiles.empty:
            save_table(timepoint_percentiles, out, "mc_timepoints")
        write_config_snapshot(config, out)
    return MonteCarloResult(metrics=metrics_df, percentiles=percentiles, timepoint_percentiles=timepoint_percentiles)


__all__ = ["run_mc", "MonteCarloResult"]
