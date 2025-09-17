import math
from pathlib import Path

from bdssim.config import load_config
from bdssim.engine.simulation import SimulationEngine


DATA_ROOT = Path('.')
BASE_CONFIG = Path('configs/base.yaml')


def run_sim(overrides: dict, horizon: int, initial_adopters=None):
    merged = {
        "objective": {"horizon_days": horizon, "residual_vol": 0.0},
        "exogenous": {"stochastic_flow_bps": 0.0},
        "policy_us": {"slippage_budget_bps": 1000, "smart_pacing": False, "max_frac_adv": 0.2},
    }
    merged.update(overrides)
    cfg = load_config(BASE_CONFIG, overrides=merged)
    engine = SimulationEngine(cfg, data_root=DATA_ROOT, initial_adopters=initial_adopters)
    result = engine.run()
    return engine, result


def test_non_us_adoption_waits_for_lag_and_progress():
    overrides = {
        "adoption": {
            "min_lag_years": 2.0,
            "progress_threshold": 0.5,
            "bass": {"p": 0.08, "q": 0.5, "m": 8},
        }
    }
    horizon = 1200
    engine, result = run_sim(overrides, horizon, initial_adopters=["USA"])
    non_us_entries = [(day, code) for day, code in result.adoption_log if code != "USA"]
    assert non_us_entries, "Expected at least one non-U.S. adoption once the gate opens"
    first_day = min(day for day, _ in non_us_entries)
    assert first_day >= math.floor(2 * 365), "Non-U.S. adoption occurred before the minimum lag"
    assert engine.us_progress >= 0.5, "U.S. progress did not reach the configured threshold"


def test_no_unlock_when_lag_exceeds_horizon():
    overrides = {
        "adoption": {
            "min_lag_years": 5.0,
            "progress_threshold": 0.25,
        }
    }
    horizon = 900
    _, result = run_sim(overrides, horizon, initial_adopters=["USA"])
    assert all(code == "USA" for _, code in result.adoption_log), "Non-U.S. adopters should remain locked"
