from pathlib import Path

import pandas as pd
import pytest

import sys
import types


class _StreamlitStub:
    def __init__(self) -> None:
        self.session_state = {}
        self.sidebar = self

    def __getattr__(self, name: str):
        if name == "columns":
            def columns(count: int):
                return [self for _ in range(count)]
            return columns
        if name == "expander":
            class _Expander:
                def __enter__(self_inner):
                    return self

                def __exit__(self_inner, exc_type, exc, tb):
                    return False
            return lambda *args, **kwargs: _Expander()
        if name == "selectbox":
            return lambda label, options, index=0, **kwargs: options[index]
        if name == "multiselect":
            return lambda *args, **kwargs: []
        if name == "slider":
            return lambda label, **kwargs: kwargs.get("value", kwargs.get("min_value", 0))
        if name == "radio":
            return lambda label, options, index=0, **kwargs: options[index]
        if name == "text_input":
            return lambda label, value="", **kwargs: value
        if name == "metric":
            return lambda *args, **kwargs: None
        if name in {"progress", "line_chart", "dataframe", "table", "markdown", "write", "warning", "info", "error", "caption", "set_page_config", "title"}:
            return lambda *args, **kwargs: None
        if name == "button":
            return lambda *args, **kwargs: False
        if name == "download_button":
            return lambda *args, **kwargs: None
        return lambda *args, **kwargs: None


sys.modules.setdefault("streamlit", _StreamlitStub())

import bdssim.dashboard_arcade as arcade  # noqa: E402
from bdssim.ceilings import COFERInputs, TAMInputs, build_ceiling_table  # noqa: E402


def test_mc_artifact_roundtrip(monkeypatch, tmp_path: Path) -> None:
    target_file = tmp_path / "dashboard_latest.json"
    monkeypatch.setattr(arcade, "MC_ARTIFACT_PATH", target_file)
    monkeypatch.setattr(arcade, "MC_ARTIFACT_DIR", tmp_path)
    payload = arcade._mc_payload(
        {"price": {"p10": 150_000.0, "p50": 200_000.0, "p90": 250_000.0}},
        {"price": [120_000.0, 140_000.0]},
        {"seed": 7, "draws": 16, "config_hash": "abc"},
    )
    arcade._persist_mc(payload)
    assert target_file.exists()
    loaded = arcade._load_latest_mc()
    assert loaded is not None
    assert loaded["percentiles"]["price"]["p50"] == 200_000.0


def test_build_ceiling_table_formats() -> None:
    df = pd.DataFrame(
        build_ceiling_table(
            TAMInputs(20e12, 0.15, 12e6),
            COFERInputs(12e12, 0.04, 12e6),
            {"p10": 150_000.0, "p50": 200_000.0, "p90": 250_000.0},
        )
    )
    assert {"metric", "value", "p10", "p50", "p90"} <= set(df.columns)
