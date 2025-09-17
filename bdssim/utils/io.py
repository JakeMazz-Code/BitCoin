"""Input/output helpers for bdssim."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yaml

from bdssim.config import Config


def ensure_directory(path: Path) -> None:
    """Create directory if it does not exist."""

    path.mkdir(parents=True, exist_ok=True)


def timestamped_dir(base: Path, prefix: str) -> Path:
    """Return a directory path suffixed with the current timestamp."""

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = base / prefix / stamp
    ensure_directory(path)
    return path


def load_timeseries(path: str | Path) -> pd.DataFrame:
    """Load a time-series CSV with a required ``day`` column."""

    df = pd.read_csv(path)
    if "day" not in df.columns:
        raise ValueError(f"Timeseries file {path} must contain a 'day' column")
    return df


def save_table(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    """Persist a dataframe as CSV and Parquet."""

    ensure_directory(out_dir)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    df.to_parquet(out_dir / f"{name}.parquet", index=False)


def write_config_snapshot(config: Config, out_dir: Path, filename: str = "config_snapshot.yaml") -> None:
    """Persist configuration as YAML for reproducibility."""

    ensure_directory(out_dir)
    with (out_dir / filename).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=False)


def load_result_tables(result_dir: Path, tables: Optional[Iterable[str]] = None) -> dict[str, pd.DataFrame]:
    """Load stored result tables from ``result_dir``."""

    tables = list(tables or ["timeseries", "summary"])
    loaded: dict[str, pd.DataFrame] = {}
    for name in tables:
        csv_path = result_dir / f"{name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing result table {csv_path}")
        loaded[name] = pd.read_csv(csv_path)
    return loaded


__all__ = [
    "ensure_directory",
    "timestamped_dir",
    "load_timeseries",
    "save_table",
    "write_config_snapshot",
    "load_result_tables",
]
