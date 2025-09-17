import subprocess
import sys
from pathlib import Path

import pandas as pd

from bdssim.analysis import compute_network_centrality, greedy_seed_selection
from bdssim.config import load_config
from bdssim.engine.simulation import SimulationEngine


def test_centrality_metrics(tmp_path: Path) -> None:
    cfg = load_config("configs/base.yaml")
    engine = SimulationEngine(cfg, data_root=".")
    result = engine.run()
    centrality = compute_network_centrality(engine.catalog)
    assert {"country", "degree", "eigenvector", "betweenness", "bloc"}.issubset(centrality.columns)
    seeds = greedy_seed_selection(engine.catalog, k=3)
    assert len(seeds) == 3
    assert len(set(seeds)) == 3
    assert result.timeseries["price"].notna().all()


def test_cli_analyze(tmp_path: Path) -> None:
    out_dir = tmp_path / "analysis"
    subprocess.check_call([
        sys.executable,
        "-m",
        "bdssim.cli",
        "analyze",
        "--config",
        "configs/base.yaml",
        "--out",
        str(out_dir),
        "--data-root",
        ".",
        "--top-k",
        "3",
    ])
    analysis_files = list(out_dir.rglob("analysis.md"))
    assert analysis_files, "analysis markdown not generated"
    md_text = analysis_files[0].read_text(encoding="utf-8")
    assert "Scenario Analytics" in md_text
