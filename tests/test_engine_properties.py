import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from bdssim.config import load_config
from bdssim.engine.simulation import SimulationEngine


def test_simulation_deterministic(tmp_path: Path) -> None:
    cfg = load_config("configs/base.yaml")
    cfg = cfg.model_copy(update={"objective": cfg.objective.model_copy(update={"horizon_days": 60})})
    engine1 = SimulationEngine(cfg)
    result1 = engine1.run()
    engine2 = SimulationEngine(cfg)
    result2 = engine2.run()
    pd.testing.assert_series_equal(result1.timeseries["price"], result2.timeseries["price"])


def test_cli_smoke(tmp_path: Path) -> None:
    cfg = load_config("configs/base.yaml")
    data = cfg.model_dump(mode="json")
    data["objective"]["horizon_days"] = 30
    cfg_path = tmp_path / "config.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    out_dir = tmp_path / "run"
    subprocess.check_call([
        sys.executable,
        "-m",
        "bdssim.cli",
        "run",
        "--config",
        str(cfg_path),
        "--out",
        str(out_dir),
        "--data-root",
        ".",
    ])
    timeseries = list(out_dir.rglob("timeseries.csv"))
    assert timeseries, "timeseries.csv not produced"
    pngs = list(out_dir.rglob("*.png"))
    assert len(pngs) >= 3
    subprocess.check_call([
        sys.executable,
        "-m",
        "bdssim.cli",
        "plot",
        "--result",
        str(timeseries[0].parent),
        "price",
        "adoption",
        "coverage",
    ])


def test_initial_adopters_seeded() -> None:
    cfg = load_config("configs/base.yaml")
    engine = SimulationEngine(cfg, initial_adopters=["CAN", "GBR"])
    assert {"CAN", "GBR"}.issubset(engine.adopted_countries)


def test_playbook_schedule_triggers() -> None:
    cfg = load_config("configs/base.yaml")
    cfg = cfg.model_copy(update={
        "objective": cfg.objective.model_copy(update={"horizon_days": 30}),
        "adoption": cfg.adoption.model_copy(update={"min_lag_years": 0.0, "progress_threshold": 0.0}),
    })
    playbook = [{"name": "North Allies", "start_day": 5, "countries": ["CAN", "GBR"]}]
    engine = SimulationEngine(cfg, playbook=playbook)
    engine.run()
    assert any(day == 5 and code == "CAN" for day, code in engine.adoption_log)
    assert "CAN" in engine.adopted_countries

