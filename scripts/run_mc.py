"""Run bdssim Monte Carlo and persist artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from bdssim.engine.mc import run_mc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bdssim Monte Carlo and store artifact.")
    parser.add_argument("config", type=Path, help="Path to JSON config overrides")
    parser.add_argument("--draws", type=int, default=64, help="Number of Monte Carlo draws")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/mc"),
        help="Output directory for Monte Carlo artifacts",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object")
    return data


def _artifact_path(base: Path, token: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{token}.json"


def _print_summary(percentiles: dict[str, dict[str, float]]) -> None:
    price = percentiles.get("price", {})
    df = pd.DataFrame([
        {"Percentile": key, "Price": value} for key, value in price.items()
    ])
    if df.empty:
        print("No percentile data available")
        return
    df.sort_values("Percentile", inplace=True)
    print(df.to_string(index=False, formatters={"Price": lambda v: f"${v:,.0f}"}))


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    result = run_mc(config, draws=args.draws, seed=args.seed)
    payload = {
        "percentiles": result.percentiles,
        "series": result.series,
        "meta": result.meta,
    }
    token = f"{result.meta['config_hash']}_{args.seed}_{args.draws}"
    artifact_path = _artifact_path(args.out, token)
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Monte Carlo percentiles (price):")
    _print_summary(result.percentiles)
    print(f"\nArtifact saved to {artifact_path}")


if __name__ == "__main__":
    main()
