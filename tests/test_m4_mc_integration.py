import pytest

from bdssim.engine.mc import run_mc
from bdssim.engine.stats import MCRun


@pytest.fixture(scope="module")
def baseline_config() -> dict:
    return {
        "years": 4,
        "start_price": 120_000,
        "start_effective_float": 16_000_000,
        "postcap_enabled": True,
        "postcap_year_index": 2,
        "alpha_postcap_delta": 0.10,
        "demand_growth_postcap_delta": 0.02,
        "hodl_reflexivity": 0.05,
        "min_tradable_float": 1_000_000,
        "noise_vol": 0.01,
    }


def test_run_mc_returns_structured_result(baseline_config: dict) -> None:
    result = run_mc(baseline_config, draws=8, seed=7)
    assert isinstance(result, MCRun)
    assert set(result.percentiles.keys()) >= {"price", "tradable_float"}
    assert set(result.series.keys()) >= {"price", "tradable_float", "reserve_value"}
    assert "years" in result.meta
    assert len(result.series["price"]) == len(result.meta["years"])


def test_percentile_keys_present(baseline_config: dict) -> None:
    result = run_mc(baseline_config, draws=8, seed=11)
    price_pct = result.percentiles["price"]
    assert set(price_pct.keys()) == {"p10", "p50", "p90"}

