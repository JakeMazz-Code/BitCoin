"""Summary table generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bdssim.utils.io import save_table


def summarize_timeseries(timeseries: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics across key metrics."""

    summary = pd.DataFrame(
        {
            "metric": [
                "final_price",
                "peak_price",
                "min_price",
                "final_debt_coverage",
                "final_us_holdings",
                "adopters_final",
            ],
            "value": [
                timeseries["price"].iloc[-1],
                timeseries["price"].max(),
                timeseries["price"].min(),
                timeseries["debt_coverage"].iloc[-1],
                timeseries["us_holdings_btc"].iloc[-1],
                timeseries["adopters"].iloc[-1] if "adopters" in timeseries.columns else float("nan"),
            ],
        }
    )
    return summary


def summary_table(timeseries: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for :func:`summarize_timeseries`."""

    return summarize_timeseries(timeseries)


def export_summary(timeseries: pd.DataFrame, out_dir: Path, name: str = "summary") -> Path:
    table = summarize_timeseries(timeseries)
    save_table(table, out_dir, name)
    return out_dir / f"{name}.csv"


__all__ = ["summarize_timeseries", "summary_table", "export_summary"]
