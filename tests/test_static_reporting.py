import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from bdssim.cli import app
from bdssim.config import load_config
from bdssim.engine.monte_carlo import run_mc
from bdssim.playbooks import get_playbook_by_key, load_playbook_index, reduce_playbook
from bdssim.reporting.debt import build_debt_projection, project_debt
from bdssim.reporting.report import build_markdown_report, scenario_report_from_mc
from bdssim.reporting.static_meta import scenario_meta_for

STATIC_CONFIGS = [
    Path("configs/static/conservative.yaml"),
    Path("configs/static/reference.yaml"),
    Path("configs/static/accelerated.yaml"),
    Path("configs/static/reference_sell.yaml"),
]


@pytest.mark.parametrize("config_path", STATIC_CONFIGS)
def test_static_config_loads_with_extended_horizon(config_path):
    cfg = load_config(config_path)
    assert cfg.objective.horizon_days == 2555
    assert cfg.policy_us.sell_rules.enabled == ("sell" in cfg.meta.name)


@pytest.mark.parametrize(
    "config_path,expected_stages",
    [
        (Path("configs/static/conservative.yaml"), 3),
        (Path("configs/static/reference.yaml"), 4),
        (Path("configs/static/accelerated.yaml"), 4),
        (Path("configs/static/reference_sell.yaml"), 5),
    ],
)
def test_static_playbook_stage_counts(config_path, expected_stages):
    cfg = load_config(config_path)
    assert len(cfg.playbook) == expected_stages


def test_static_metadata_resolution_matches_playbook():
    meta = scenario_meta_for("reference")
    assert meta is not None
    assert meta.playbook_key == "reference_wave"
    playbook = get_playbook_by_key(meta.playbook_key)
    assert playbook is not None
    assert "BIS" in " ".join(playbook.sources)


def test_reduce_playbook_retains_probability_fields():
    playbook = get_playbook_by_key("reference_wave")
    waves = reduce_playbook(playbook)
    assert any("probability" in wave for wave in waves)


def test_project_debt_compounding_behaviour():
    projected = project_debt(1_000_000_000, 0.05, 5)
    assert pytest.approx(projected, rel=1e-6) == 1_000_000_000 * (1.05**5)


def test_build_debt_projection_returns_expected_years():
    rows = build_debt_projection(100, 0.1, [2, 5, 7])
    assert [row.years for row in rows] == [2, 5, 7]


def test_run_mc_includes_timepoint_percentiles(tmp_path):
    cfg = load_config(Path("configs/static/conservative.yaml"))
    result = run_mc(cfg, runs=2)
    assert not result.timepoint_percentiles.empty
    assert set(result.timepoint_percentiles["years"].unique()) == {2, 5, 7}


def test_report_includes_checkpoint_section(tmp_path):
    cfg = load_config(Path("configs/static/reference.yaml"))
    mc = run_mc(cfg, runs=2)
    report = scenario_report_from_mc(
        config=cfg,
        metrics=mc.metrics,
        percentiles=mc.percentiles,
        timepoint_percentiles=mc.timepoint_percentiles,
    )
    markdown = build_markdown_report([report])
    assert "Holdings vs debt checkpoints" in markdown


def test_report_includes_playbook_sources():
    cfg = load_config(Path("configs/static/accelerated.yaml"))
    mc = run_mc(cfg, runs=2)
    meta = scenario_meta_for(cfg.meta.name)
    playbook = get_playbook_by_key(meta.playbook_key) if meta else None
    report = scenario_report_from_mc(
        config=cfg,
        metrics=mc.metrics,
        percentiles=mc.percentiles,
        timepoint_percentiles=mc.timepoint_percentiles,
        playbook=playbook,
    )
    markdown = build_markdown_report([report])
    assert "Adoption wave schedule" in markdown
    assert "Sources" in markdown


@pytest.mark.parametrize("key", ["conservative_wave", "reference_wave", "accelerated_wave"])
def test_playbook_index_contains_static_entries(key):
    entries = {pb.key for pb in load_playbook_index()}
    assert key in entries


def test_static_cli_report_runs(tmp_path):
    runner = CliRunner()
    out_dir = tmp_path / "reports"
    result = runner.invoke(
        app,
        [
            "report",
            "--runs",
            "10",
            "--out",
            str(out_dir),
            str(Path("configs/static/conservative.yaml")),
            str(Path("configs/static/reference.yaml")),
        ],
    )
    assert result.exit_code == 0, result.output
    generated = list(out_dir.glob("*.md"))
    assert generated, "Expected markdown report to be generated"


def test_sell_configuration_enables_discretionary_rules():
    cfg = load_config(Path("configs/static/reference_sell.yaml"))
    assert cfg.policy_us.sell_rules.enabled is True
    assert cfg.policy_us.sell_rules.time_lock_days <= 20


def test_static_metadata_contains_assumptions():
    meta = scenario_meta_for("accelerated")
    assert meta and len(meta.key_assumptions) >= 1


def test_timepoint_percentiles_structure():
    cfg = load_config(Path("configs/static/reference.yaml"))
    mc = run_mc(cfg, runs=2)
    columns = {"years", "percentile", "day", "price", "debt_coverage", "us_holdings_btc"}
    assert columns.issubset(mc.timepoint_percentiles.columns)


def test_playbook_sources_present_in_index():
    playbook = get_playbook_by_key("accelerated_wave")
    assert playbook is not None
    assert playbook.sources


def test_static_metadata_debt_source_strings():
    for key in ("conservative", "reference", "accelerated", "reference_sell"):
        meta = scenario_meta_for(key)
        assert meta is not None
        assert "CBO" in meta.debt_source or "IMF" in meta.debt_source
