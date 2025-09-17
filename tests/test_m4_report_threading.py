import matplotlib
matplotlib.use("Agg")

import pandas as pd
from pathlib import Path

from bdssim.engine.stats import MCRun
from bdssim.reporting.report_builder import build_mc_sections


def _mock_mc_run() -> MCRun:
    percentiles = {
        "price": {"p10": 150_000.0, "p50": 200_000.0, "p90": 260_000.0},
        "tradable_float": {"p10": 13_000_000.0, "p50": 12_000_000.0, "p90": 11_500_000.0},
        "reflexivity_delta": {"p10": 5_000.0, "p50": 8_000.0, "p90": 12_000.0},
    }
    series = {
        "price": [120_000.0, 150_000.0, 180_000.0],
        "tradable_float": [16_000_000.0, 14_500_000.0, 12_500_000.0],
        "reserve_value": [1.9e12, 2.1e12, 2.3e12],
    }
    meta = {"years": [0, 1, 2], "seed": 7, "draws": 16, "runtime_sec": 0.1, "config_hash": "abc123"}
    return MCRun(percentiles=percentiles, series=series, meta=meta)


def test_report_includes_sections(tmp_path: Path) -> None:
    mc = _mock_mc_run()
    meta = {
        "postcap_year_index": 1,
        "alpha_postcap_delta": 0.12,
        "demand_growth_postcap_delta": 0.03,
        "tam_inputs": {"tam_usd": 20e12, "btc_share": 0.15, "tradable_supply": 12e6},
        "cofer_inputs": {"cofer_usd": 12e12, "reserve_share": 0.04, "tradable_supply": 12e6},
    }
    ceilings_df = pd.DataFrame(
        {
            "metric": ["TAM share implied price", "COFER share implied price"],
            "value": [250_000.0, 220_000.0],
            "p10": [150_000.0, 150_000.0],
            "p50": [180_000.0, 180_000.0],
            "p90": [230_000.0, 230_000.0],
        }
    )
    markdown = build_mc_sections(mc=mc, meta=meta, ceilings_df=ceilings_df, out_dir=tmp_path)
    assert "Post-Cap Monetization" in markdown
    assert "How High Could It Go?" in markdown
    assert "TAM lens" in markdown
    assert "COFER lens" in markdown
    assert "Model horizon price" in markdown
    assert (tmp_path / "postcap_bend_mc.png").exists()
