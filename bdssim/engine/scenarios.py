"""Scenario helper utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from bdssim.config import Config, load_config


SCENARIO_FILES: Dict[str, str] = {
    "base": "base.yaml",
    "optimistic": "optimistic.yaml",
    "stress_liquidity": "stress_liquidity.yaml",
    "threshold_network": "threshold_network.yaml",
}


def load_scenario(name: str, config_dir: str | Path = "configs") -> Config:
    """Load a scenario configuration by name."""

    rel = SCENARIO_FILES.get(name, name)
    path = Path(config_dir) / rel
    return load_config(path)


__all__ = ["load_scenario", "SCENARIO_FILES"]
