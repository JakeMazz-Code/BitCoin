import importlib
import numpy as np
import pandas as pd


def _get_sim_run():
    sim = importlib.import_module("bdssim.engine.simulation")
    for name in ("run_years", "run", "simulate"):
        fn = getattr(sim, name, None)
        if callable(fn):
            return fn
    raise AssertionError("No simulation runner found (expected run_years/run/simulate).")


def test_postcap_monotonic_when_positive_drift() -> None:
    SIM_RUN = _get_sim_run()
    cfg = {
        "years": 6,
        "postcap_enabled": True,
        "postcap_year_index": 3,
        "alpha_postcap_delta": 0.10,
        "demand_growth_postcap_delta": 0.02,
        "hodl_reflexivity": 0.10,
        "noise_vol": 0.0,
        "min_tradable_float": 1_000_000,
        "start_effective_float": 16_000_000,
        "start_price": 115_000,
    }
    df = SIM_RUN(config=cfg)
    pc = cfg["postcap_year_index"]
    post = df[df["year"] >= pc]["price"].values
    assert np.all(post[1:] >= post[:-1])


def test_reflexivity_tightens_float_and_logs_observability() -> None:
    SIM_RUN = _get_sim_run()
    cfg_on = {
        "years": 6,
        "postcap_enabled": True,
        "postcap_year_index": 3,
        "alpha_postcap_delta": 0.10,
        "demand_growth_postcap_delta": 0.02,
        "hodl_reflexivity": 0.10,
        "noise_vol": 0.0,
        "min_tradable_float": 1_000_000,
        "start_effective_float": 16_000_000,
        "start_price": 115_000,
    }
    cfg_off = {**cfg_on, "hodl_reflexivity": 0.0}
    df_on = SIM_RUN(config=cfg_on)
    df_off = SIM_RUN(config=cfg_off)
    year_max = df_on["year"].max()
    tf_on = df_on.loc[df_on["year"] == year_max, "tradable_float"].iloc[0]
    tf_off = df_off.loc[df_off["year"] == year_max, "tradable_float"].iloc[0]
    assert tf_on <= tf_off
    for col in ("prior_year_return", "reflexivity_delta"):
        assert col in df_on.columns
        assert pd.notna(df_on.loc[df_on["year"] == year_max, col]).any()


def test_hard_cap_and_min_float_invariants() -> None:
    SIM_RUN = _get_sim_run()
    cfg = {
        "years": 6,
        "postcap_enabled": True,
        "postcap_year_index": 3,
        "alpha_postcap_delta": 0.10,
        "demand_growth_postcap_delta": 0.02,
        "hodl_reflexivity": 0.10,
        "noise_vol": 0.0,
        "min_tradable_float": 1_000_000,
        "start_effective_float": 16_000_000,
        "start_price": 115_000,
    }
    df = SIM_RUN(config=cfg)
    assert (df["circ_supply"] <= 21_000_000 + 1e-6).all()
    assert (df["tradable_float"] >= cfg["min_tradable_float"]).all()
