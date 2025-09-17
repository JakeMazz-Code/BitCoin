import pytest

from bdssim.engine.mc import run_mc


@pytest.fixture(scope="module")
def mc_result() -> object:
    config = {
        "years": 5,
        "start_price": 110_000,
        "start_effective_float": 15_000_000,
        "postcap_enabled": True,
        "postcap_year_index": 2,
        "alpha_postcap_delta": 0.15,
        "demand_growth_postcap_delta": 0.03,
        "hodl_reflexivity": 0.08,
        "min_tradable_float": 900_000,
        "noise_vol": 0.015,
    }
    return run_mc(config, draws=16, seed=21)


def test_percentiles_monotonicity(mc_result) -> None:
    p = mc_result.percentiles["price"]
    assert p["p10"] <= p["p50"] <= p["p90"]


def test_mean_coverage(mc_result) -> None:
    horizon_mean = mc_result.series["price"][-1]
    analytic = mc_result.percentiles["price"]["p50"]
    rel_error = abs(horizon_mean - analytic) / max(abs(analytic), 1e-9)
    assert rel_error < 0.1

