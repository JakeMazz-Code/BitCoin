"""Streamlit dashboard for interactive bdssim exploration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from bdssim.config import Config, load_config
from bdssim.data.market_data import fetch_spot_price
from bdssim.engine.simulation import SimulationEngine
from bdssim.engine.scenarios import SCENARIO_FILES
from bdssim.reporting.summary import summary_table

DATA_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = DATA_ROOT / "configs"
SESSION_PRICE_KEY = "bdssim_live_price"


@st.cache_data(show_spinner=False)
def run_simulation(config_dict: Dict[str, Any], data_root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the simulation with ``config_dict`` and return timeseries + summary."""

    cfg = Config.model_validate(config_dict)
    engine = SimulationEngine(cfg, data_root=data_root)
    result = engine.run()
    return result.timeseries, summary_table(result.timeseries)


def _scenario_path(name: str) -> Path:
    file_name = SCENARIO_FILES.get(name, name)
    return CONFIG_ROOT / file_name


def _load_base_config(name: str) -> Config:
    return load_config(_scenario_path(name))


def _apply_sidebar_controls(cfg: Config) -> Config:
    if SESSION_PRICE_KEY not in st.session_state:
        st.session_state[SESSION_PRICE_KEY] = float(cfg.market.init_price_usd)

    st.sidebar.header("Market")
    horizon = st.sidebar.slider(
        "Horizon (days)",
        min_value=90,
        max_value=3650,
        value=int(cfg.objective.horizon_days),
        step=30,
    )
    price_placeholder = st.session_state[SESSION_PRICE_KEY]
    if st.sidebar.button("Use live BTC price", help="Fetch spot price via CoinGecko"):
        try:
            live_price = fetch_spot_price()
            st.session_state[SESSION_PRICE_KEY] = live_price
            st.sidebar.success(f"${live_price:,.0f}")
        except Exception as exc:  # pragma: no cover - UI feedback only
            st.sidebar.error(f"Price fetch failed: {exc}")
    price_input = st.sidebar.number_input(
        "Initial BTC price (USD)",
        min_value=1_000.0,
        max_value=500_000.0,
        value=float(st.session_state[SESSION_PRICE_KEY]),
        step=1_000.0,
        format="%.0f",
    )
    st.session_state[SESSION_PRICE_KEY] = price_input

    adv = st.sidebar.number_input(
        "Average daily volume (USD billions)",
        min_value=1.0,
        max_value=50.0,
        value=cfg.market.adv_usd / 1e9,
        step=1.0,
    )
    sigma = st.sidebar.slider("Daily volatility", min_value=0.01, max_value=0.15, value=float(cfg.market.init_sigma), step=0.005)

    st.sidebar.header("Impact & Execution")
    y_buy = st.sidebar.slider("Y buy", min_value=0.1, max_value=2.0, value=float(cfg.impact.Y_buy), step=0.05)
    y_sell = st.sidebar.slider("Y sell", min_value=0.1, max_value=2.5, value=float(cfg.impact.Y_sell), step=0.05)
    phi = st.sidebar.slider("Permanent impact fraction", min_value=0.0, max_value=0.6, value=float(cfg.impact.permanent_phi_buy), step=0.05)
    tau = st.sidebar.slider("Impact decay tau (days)", min_value=1.0, max_value=15.0, value=float(cfg.impact.temp_decay_tau_days), step=0.5)

    usd_per_day = st.sidebar.number_input(
        "US purchases (USD billions / day)",
        min_value=0.0,
        max_value=10.0,
        value=cfg.policy_us.usd_per_day / 1e9,
        step=0.1,
    )
    adv_cap = st.sidebar.slider("Max % ADV", min_value=0.0, max_value=0.3, value=float(cfg.policy_us.max_frac_adv), step=0.01)

    adoption = cfg.adoption
    if adoption.model == "bass":
        st.sidebar.header("Bass adoption")
        bass = adoption.bass
        p = st.sidebar.slider("Innovation (p)", min_value=0.0, max_value=0.1, value=float(bass.p), step=0.001)
        q = st.sidebar.slider("Imitation (q)", min_value=0.0, max_value=1.0, value=float(bass.q), step=0.01)
        m = st.sidebar.slider("Market potential", min_value=5, max_value=50, value=int(bass.m), step=1)
        adoption = adoption.model_copy(update={"bass": bass.model_copy(update={"p": p, "q": q, "m": m})})
    else:
        st.sidebar.header("Threshold adoption")
        threshold = adoption.threshold
        default_theta = st.sidebar.slider("Base threshold", min_value=0.3, max_value=0.8, value=float(threshold.default_theta), step=0.01)
        alpha_peer = st.sidebar.slider("Peer sensitivity", min_value=0.0, max_value=0.6, value=float(threshold.alpha_peer), step=0.05)
        policy_penalty = st.sidebar.slider("Policy penalty", min_value=0.0, max_value=0.3, value=float(threshold.policy_penalty), step=0.02)
        adoption = adoption.model_copy(
            update={
                "threshold": threshold.model_copy(
                    update={
                        "default_theta": default_theta,
                        "alpha_peer": alpha_peer,
                        "policy_penalty": policy_penalty,
                    }
                )
            }
        )

    market = cfg.market.model_copy(update={"adv_usd": adv * 1e9, "init_sigma": sigma, "init_price_usd": price_input})
    impact = cfg.impact.model_copy(update={
        "Y_buy": y_buy,
        "Y_sell": y_sell,
        "permanent_phi_buy": phi,
        "permanent_phi_sell": phi,
        "temp_decay_tau_days": tau,
    })
    policy = cfg.policy_us.model_copy(update={
        "usd_per_day": usd_per_day * 1e9,
        "max_frac_adv": adv_cap,
    })
    objective = cfg.objective.model_copy(update={"horizon_days": horizon})
    meta = cfg.meta.model_copy(update={"name": f"{cfg.meta.name}-custom"})

    return cfg.model_copy(update={
        "meta": meta,
        "market": market,
        "impact": impact,
        "policy_us": policy,
        "adoption": adoption,
        "objective": objective,
    })


def _display_outputs(timeseries: pd.DataFrame, summary: pd.DataFrame) -> None:
    st.subheader("Summary statistics")
    st.dataframe(summary, hide_index=True, use_container_width=True)

    st.subheader("Price path")
    st.line_chart(timeseries.set_index("day")["price"], use_container_width=True)

    st.subheader("Debt coverage")
    st.line_chart(timeseries.set_index("day")["debt_coverage"], use_container_width=True)

    st.subheader("Adoption count")
    if "adopters" in timeseries.columns:
        st.line_chart(timeseries.set_index("day")["adopters"], use_container_width=True)

    csv_bytes = timeseries.to_csv(index=False).encode("utf-8")
    st.download_button("Download timeseries CSV", data=csv_bytes, file_name="bdssim_timeseries.csv", mime="text/csv")


def main() -> None:
    st.set_page_config(page_title="BDSSim Dashboard", layout="wide")
    st.title("Bitcoin Debt Solution Simulator")
    st.markdown("Adjust parameters in the sidebar and explore outcomes in real time. Fetch live BTC prices on demand or use scenario presets for counterfactual analysis.")

    scenario_names = list(SCENARIO_FILES.keys())
    selected = st.sidebar.selectbox("Scenario", scenario_names, index=scenario_names.index("base"))
    base_cfg = _load_base_config(selected)
    tuned_cfg = _apply_sidebar_controls(base_cfg)

    config_dict = tuned_cfg.model_dump(mode="json")
    with st.spinner("Running simulation..."):
        timeseries, summary = run_simulation(config_dict, str(DATA_ROOT))
    _display_outputs(timeseries, summary)


if __name__ == "__main__":
    main()
