import pandas as pd
from pathlib import Path

from bdssim.reporting.report_builder import build_markdown_report


def test_report_contains_postcap_and_how_high_sections(tmp_path: Path) -> None:
    sim_df = pd.DataFrame({
        "year": [0, 1, 2, 3, 4, 5],
        "price": [115_000, 120_000, 130_000, 140_000, 160_000, 180_000],
        "tradable_float": [16e6, 15.8e6, 15.5e6, 15.0e6, 14.2e6, 13.6e6],
        "prior_year_return": [float("nan"), 0.043, 0.083, 0.077, 0.143, 0.125],
        "reflexivity_delta": [0.0, 0.0, 0.0, 10_000.0, 12_000.0, 14_000.0],
    })
    meta = {
        "postcap_enabled": True,
        "postcap_year_index": 3,
        "alpha_postcap_delta": 0.10,
        "demand_growth_postcap_delta": 0.02,
    }
    ceilings_df = pd.DataFrame({
        "metric": ["TAM share implied price", "COFER share implied price"],
        "value": [250_000.0, 220_000.0],
        "p10": [150_000.0, 150_000.0],
        "p50": [180_000.0, 180_000.0],
        "p90": [230_000.0, 230_000.0],
    })
    out = build_markdown_report(sim_df=sim_df, meta=meta, ceilings_df=ceilings_df, out_dir=tmp_path)
    text = Path(out).read_text(encoding="utf-8")
    assert "Post-Cap Monetization" in text
    assert "How High Could It Go?" in text
    assert "alpha" in text.lower()
    assert "demand" in text.lower()
