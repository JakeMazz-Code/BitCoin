import numpy as np
import pandas as pd
from pathlib import Path

from bdssim.config import load_config
from bdssim.engine.simulation import SimulationEngine

DATA_ROOT = Path('.')
BASE_CONFIG = Path('configs/base.yaml')


def run_sim(overrides: dict, horizon: int):
    base = {
        "objective": {"horizon_days": horizon, "residual_vol": 0.0},
        "exogenous": {"stochastic_flow_bps": 0.0},
        "policy_us": {
            "slippage_budget_bps": 1000,
            "smart_pacing": False,
            "max_frac_adv": 0.2,
        },
        "adoption": {
            "min_lag_years": 0.0,
            "progress_threshold": 0.0,
        },
    }
    merged = base
    merged.update(overrides)
    cfg = load_config(BASE_CONFIG, overrides=merged)
    engine = SimulationEngine(cfg, data_root=DATA_ROOT, initial_adopters=["USA"])
    result = engine.run()
    return engine, result.timeseries


def test_postcap_drift_pushes_price_up():
    overrides = {
        "market": {"scarcity_alpha": 0.0, "demand_growth": 0.0},
        "postcap": {
            "enabled": True,
            "switch_year_index": 1,
            "alpha_postcap_delta": 0.0,
            "demand_growth_postcap_delta": 0.05,
            "hodl_reflexivity": 0.0,
            "min_tradable_float": 500_000.0,
        },
    }
    horizon = 6 * 365
    _, ts = run_sim(overrides, horizon)
    switch_day = overrides["postcap"]["switch_year_index"] * 365
    prices = ts["price"].values
    for idx in range(switch_day + 1, len(prices)):
        assert prices[idx] > prices[idx - 1]
    assert prices[-1] > prices[switch_day]


def test_reflexivity_tightens_float():
    common_overrides = {
        "market": {"scarcity_alpha": 0.0, "demand_growth": 0.0},
        "postcap": {
            "enabled": True,
            "switch_year_index": 1,
            "alpha_postcap_delta": 0.05,
            "demand_growth_postcap_delta": 0.05,
            "min_tradable_float": 500_000.0,
        },
    }
    overrides_loose = {
        "postcap": {
            **common_overrides["postcap"],
            "hodl_reflexivity": 0.0,
        }
    }
    overrides_tight = {
        "postcap": {
            **common_overrides["postcap"],
            "hodl_reflexivity": 0.2,
        }
    }
    horizon = 6 * 365
    _, ts_loose = run_sim({**common_overrides, **overrides_loose}, horizon)
    _, ts_tight = run_sim({**common_overrides, **overrides_tight}, horizon)
    assert ts_tight["effective_float"].iloc[-1] < ts_loose["effective_float"].iloc[-1]


def test_reflexivity_respects_min_float():
    overrides = {
        "market": {"scarcity_alpha": 0.0, "demand_growth": 0.0},
        "postcap": {
            "enabled": True,
            "switch_year_index": 1,
            "alpha_postcap_delta": 0.1,
            "demand_growth_postcap_delta": 0.05,
            "hodl_reflexivity": 1.0,
            "min_tradable_float": 5_000_000.0,
        },
    }
    _, ts = run_sim(overrides, 6 * 365)
    assert ts["effective_float"].min() >= overrides["postcap"]["min_tradable_float"]


def test_circulating_supply_respects_cap():
    overrides = {}
    _, ts = run_sim(overrides, 4 * 365)
    assert ts["circulating_supply"].max() <= 21_000_000 + 1e-6


def test_postcap_zero_deltas_matches_baseline():
    base_overrides = {
        "postcap": {
            "enabled": False,
        },
    }
    zero_delta = {
        "postcap": {
            "enabled": True,
            "switch_year_index": 1,
            "alpha_postcap_delta": 0.0,
            "demand_growth_postcap_delta": 0.0,
            "hodl_reflexivity": 0.0,
        },
    }
    horizon = 4 * 365
    _, ts_disabled = run_sim(base_overrides, horizon)
    _, ts_zero = run_sim(zero_delta, horizon)
    pd.testing.assert_series_equal(ts_disabled["price"], ts_zero["price"], check_names=False)
    pd.testing.assert_series_equal(ts_disabled["effective_float"], ts_zero["effective_float"], check_names=False)


def test_alpha_delta_elevates_price():
    overrides_base = {
        "market": {"scarcity_alpha": 0.0},
        "postcap": {
            "enabled": True,
            "switch_year_index": 1,
            "alpha_postcap_delta": 0.0,
            "demand_growth_postcap_delta": 0.02,
            "hodl_reflexivity": 0.0,
        },
    }
    overrides_alpha = {
        "market": {"scarcity_alpha": 0.0},
        "postcap": {
            "enabled": True,
            "switch_year_index": 1,
            "alpha_postcap_delta": 0.2,
            "demand_growth_postcap_delta": 0.02,
            "hodl_reflexivity": 0.0,
        },
    }
    horizon = 6 * 365
    _, ts_base = run_sim(overrides_base, horizon)
    _, ts_alpha = run_sim(overrides_alpha, horizon)
    assert ts_alpha["price"].iloc[-1] > ts_base["price"].iloc[-1]


def test_demand_delta_elevates_price():
    overrides_base = {
        "postcap": {
            "enabled": True,
            "switch_year_index": 1,
            "alpha_postcap_delta": 0.0,
            "demand_growth_postcap_delta": 0.0,
            "hodl_reflexivity": 0.0,
        },
    }
    overrides_growth = {
        "postcap": {
            "enabled": True,
            "switch_year_index": 1,
            "alpha_postcap_delta": 0.0,
            "demand_growth_postcap_delta": 0.05,
            "hodl_reflexivity": 0.0,
        },
    }
    horizon = 6 * 365
    _, ts_base = run_sim(overrides_base, horizon)
    _, ts_growth = run_sim(overrides_growth, horizon)
    assert ts_growth["price"].iloc[-1] > ts_base["price"].iloc[-1]


def test_postcap_runs_are_deterministic():
    overrides = {
        "postcap": {
            "enabled": True,
            "switch_year_index": 1,
            "alpha_postcap_delta": 0.1,
            "demand_growth_postcap_delta": 0.02,
            "hodl_reflexivity": 0.1,
        },
    }
    horizon = 4 * 365
    _, ts1 = run_sim(overrides, horizon)
    _, ts2 = run_sim(overrides, horizon)
    pd.testing.assert_frame_equal(ts1.reset_index(drop=True), ts2.reset_index(drop=True))
