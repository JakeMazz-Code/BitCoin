"""Gamified Streamlit dashboard for bdssim."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json

import pandas as pd
import streamlit as st

from bdssim.analysis import compute_network_centrality, summarize_cascade, yearly_summary
from bdssim.adoption.countries import CountryCatalog, load_countries, load_country_graph
from bdssim.config import Config, load_config
from bdssim.data.market_data import fetch_spot_price
from bdssim.engine.mc import run_mc
from bdssim.engine.simulation import SimulationEngine
from bdssim.playbooks import Playbook, load_playbook_index, reduce_playbook
from bdssim.ceilings import COFERInputs, TAMInputs, build_ceiling_table

DATA_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = DATA_ROOT / "configs"
PLAYBOOK_INDEX_PATH = CONFIG_ROOT / "playbooks" / "index.yaml"
MISSION_KEY = "bdssim_mission_state"
MC_ARTIFACT_DIR = DATA_ROOT / "artifacts" / "mc"
MC_ARTIFACT_PATH = MC_ARTIFACT_DIR / "dashboard_latest.json"

BLOC_PRESETS = {
    "North Atlantic": ["CAN", "GBR"],
    "Indo-Pacific Partners": ["JPN", "KOR", "AUS"],
    "Continental Europe": ["DEU", "FRA", "ITA"],
    "North America": ["CAN", "MEX"],
    "BRICS Core": ["CHN", "IND", "BRA", "RUS", "ZAF"],
    "Energy Exporters": ["SAU"],
    "Emerging Allies": ["TUR", "ARG", "IDN"],
}

DEFAULT_CUSTOM = [
    {"name": "Phase 1", "start_day": 0, "blocs": ["North Atlantic"]},
    {"name": "Phase 2", "start_day": 200, "blocs": ["Continental Europe"]},
    {"name": "Phase 3", "start_day": 400, "blocs": ["BRICS Core"]},
]


def _load_latest_mc() -> Optional[dict]:
    try:
        with MC_ARTIFACT_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    return None


def _persist_mc(payload: dict) -> None:
    MC_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MC_ARTIFACT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _mc_payload(percentiles: dict, series: dict, meta: dict) -> dict:
    return {"percentiles": percentiles, "series": series, "meta": meta}


@lru_cache(maxsize=8)
def _cached_catalog(country_csv: str, graph_csv: str) -> CountryCatalog:
    countries = load_countries(Path(country_csv))
    graph = load_country_graph(Path(graph_csv), [c.code for c in countries])
    return CountryCatalog(countries, graph)


def load_catalog(cfg: Config) -> CountryCatalog:
    country_csv = DATA_ROOT / cfg.countries.country_csv
    graph_csv = DATA_ROOT / cfg.countries.graph_csv
    try:
        return _cached_catalog(str(country_csv), str(graph_csv))
    except FileNotFoundError as exc:  # pragma: no cover - surfaced in UI
        st.sidebar.error(f"Country data not found: {exc}")
        raise


@lru_cache(maxsize=1)
def _cached_playbooks(index_path: str) -> Tuple[List[Playbook], Dict[str, Playbook]]:
    playbooks = load_playbook_index(index_path)
    return playbooks, {pb.key: pb for pb in playbooks}


def load_playbooks(index_path: Path) -> Tuple[List[Playbook], Dict[str, Playbook]]:
    try:
        return _cached_playbooks(str(index_path))
    except FileNotFoundError:  # pragma: no cover - surfaced in UI
        return [], {}


def run_mission(
    cfg: Config,
    allies: List[str],
    playbook: List[dict],
) -> tuple[pd.DataFrame, CountryCatalog, SimulationEngine]:
    engine = SimulationEngine(cfg, data_root=DATA_ROOT, initial_adopters=allies, playbook=playbook)
    result = engine.run()
    return result.timeseries, engine.catalog, engine


def parameter_help() -> None:
    st.sidebar.markdown("### How the knobs work")
    st.sidebar.write("**Mission length**: days until the simulation stops.")
    st.sidebar.write("**US daily budget**: USD spent on BTC each day. Higher budgets accumulate faster but move price more.")
    st.sidebar.write("**Market footprint**: share of global liquidity you absorb. Keep it modest to avoid slippage.")
    st.sidebar.write("**Innovation spark (p)**: chance that a country adopts on its own. Increase to seed the cascade.")
    st.sidebar.write("**Peer contagion (q)**: imitation strength. Higher values trigger faster domino effects.")
    st.sidebar.write("Toggle profit-taking if you want the US to keep stacking without selling.")
    st.sidebar.write("Use stages to schedule blocs that join later in the mission.")


def mission_score(coverage: float, adopters: int, dent: float) -> int:
    return int(min((coverage + dent) * 5000 + adopters * 4, 100))


def expand_stage(blocs: List[str], valid_codes: List[str]) -> List[str]:
    codes = set()
    for bloc in blocs:
        for code in BLOC_PRESETS.get(bloc, []):
            if code in valid_codes:
                codes.add(code)
    return sorted(codes)


def main() -> None:
    st.set_page_config(page_title="BDSSim Mission Control", layout="wide")
    st.title("??? Mission Control: Bitcoin Sovereign Strategy")
    st.markdown("Recruit allies, stage regional waves, and track how the portfolio evolves year by year.")

    scenario_files = sorted(p.stem for p in CONFIG_ROOT.glob("*.yaml"))
    if not scenario_files:
        st.error("No scenario configurations found under configs/. Add a YAML file to proceed.")
        return
    default_index = scenario_files.index("base") if "base" in scenario_files else 0
    scenario_name = st.sidebar.selectbox("Scenario preset", scenario_files, index=default_index)
    cfg = load_config(CONFIG_ROOT / f"{scenario_name}.yaml")
    catalog = load_catalog(cfg)

    mission_state = st.session_state.setdefault(MISSION_KEY, {})
    if "mc_artifact" not in mission_state:
        previous = _load_latest_mc()
        if previous:
            mission_state["mc_artifact"] = previous

    playbooks, playbook_map = load_playbooks(PLAYBOOK_INDEX_PATH)
    if not playbooks:
        st.sidebar.warning("No playbooks found. Only custom staging is available.")

    parameter_help()

    sim_mode = st.sidebar.radio("Simulation mode", ["Deterministic", "Monte Carlo"], index=0)
    mc_draws = None
    if sim_mode == "Monte Carlo":
        mc_draws = st.sidebar.slider("MC draws", min_value=8, max_value=128, value=32, step=8)

    if st.sidebar.button("Use live BTC price"):
        try:
            live_price = fetch_spot_price()
            mission_state["price"] = live_price
            st.sidebar.success(f"Price locked at ${live_price:,.0f}")
        except Exception as exc:  # pragma: no cover - UI feedback only
            st.sidebar.error(f"Price fetch failed: {exc}")
    price_default = mission_state.get("price", cfg.market.init_price_usd)

    horizon = st.sidebar.slider("Mission length (days)", 90, 3650, cfg.objective.horizon_days, 30)
    spend = st.sidebar.slider("US daily budget (USD billions)", 0.1, 5.0, cfg.policy_us.usd_per_day / 1e9, 0.1)
    footprint = st.sidebar.slider("Market footprint (% of daily liquidity)", 0.0, 0.20, float(cfg.policy_us.max_frac_adv), 0.01)
    p_coef = st.sidebar.slider("Innovation spark (p)", 0.0, 0.10, float(cfg.adoption.bass.p), 0.002)
    q_coef = st.sidebar.slider("Peer contagion (q)", 0.0, 0.90, float(cfg.adoption.bass.q), 0.02)
    profit_enabled = st.sidebar.checkbox(
        "Enable profit-taking",
        value=cfg.policy_us.sell_rules.enabled,
        help="Uncheck to accumulate without discretionary sells",
    )

    allies = st.sidebar.multiselect(
        "Stage 0 allies (start immediately)",
        options=catalog.codes(),
        default=[code for code in ("CAN", "GBR", "JPN") if code in catalog.codes()],
    )

    preset_keys = ["custom"] + [pb.key for pb in playbooks]
    quick_plan = st.sidebar.selectbox(
        "Playbook template",
        options=preset_keys,
        index=0,
        format_func=lambda key: playbook_map[key].name if key in playbook_map else "Custom",
    )
    if quick_plan != "custom" and quick_plan in playbook_map:
        selected_pb: Optional[Playbook] = playbook_map[quick_plan]
        info = selected_pb.description or ""
        if selected_pb.probability is not None:
            info = f"{info} (weight {selected_pb.probability:.0%})" if info else f"weight {selected_pb.probability:.0%}"
        if info:
            st.sidebar.info(info)
        template = reduce_playbook(selected_pb)
    else:
        selected_pb = None
        template = DEFAULT_CUSTOM

    stages_config: List[dict] = []
    catalog_codes = set(catalog.codes())
    code_list = sorted(catalog_codes)
    for idx, stage in enumerate(template):
        with st.sidebar.expander(f"Stage {idx + 1}", expanded=(idx == 0)):
            name = st.text_input(
                f"Stage {idx + 1} name",
                value=stage.get("name", f"Stage {idx + 1}"),
                key=f"stage_name_{idx}",
            )
            day = st.slider(
                f"Launch day",
                min_value=0,
                max_value=int(horizon),
                value=int(stage.get("start_day", idx * 180)),
                step=30,
                key=f"stage_day_{idx}",
            )
            default_blocs = [b for b in stage.get("blocs", []) if b in BLOC_PRESETS]
            default_countries = [c for c in stage.get("countries", []) if c in catalog_codes]
            blocs = st.multiselect(
                f"Bloc selection",
                options=list(BLOC_PRESETS.keys()),
                default=default_blocs,
                key=f"stage_blocs_{idx}",
            )
            if default_countries and not default_blocs:
                st.caption(f"Playbook countries: {', '.join(default_countries)}")
        countries = expand_stage(blocs, code_list)
        if not countries and default_countries:
            countries = sorted(set(default_countries))
        if countries:
            stage_cfg = {"name": name or f"Stage {idx + 1}", "start_day": day, "countries": countries}
            if blocs:
                stage_cfg["blocs"] = blocs
            stages_config.append(stage_cfg)

    sell_rules = cfg.policy_us.sell_rules.model_copy()
    if profit_enabled:
        sell_rules = sell_rules.model_copy(update={"enabled": True})
    else:
        sell_rules = sell_rules.model_copy(update={"enabled": False, "take_profit_bands": []})

    tuned_cfg = cfg.model_copy(
        update={
            "market": cfg.market.model_copy(update={"init_price_usd": price_default}),
            "objective": cfg.objective.model_copy(update={"horizon_days": horizon}),
            "policy_us": cfg.policy_us.model_copy(
                update={"usd_per_day": spend * 1e9, "max_frac_adv": footprint, "sell_rules": sell_rules}
            ),
            "adoption": cfg.adoption.model_copy(update={"bass": cfg.adoption.bass.model_copy(update={"p": p_coef, "q": q_coef})}),
        }
    )

    timeseries, catalog, engine = run_mission(tuned_cfg, allies, stages_config)
    cascade = summarize_cascade(timeseries)
    coverage_final = float(timeseries["debt_coverage"].iloc[-1])
    adopters_final = float(timeseries["adopters"].iloc[-1])
    final_price = float(timeseries["price"].iloc[-1])
    holdings_final = float(timeseries["us_holdings_btc"].iloc[-1])
    mark_to_market = holdings_final * final_price
    dent_pct = mark_to_market / cfg.objective.debt_baseline_usd
    score = mission_score(coverage_final, adopters_final, dent_pct)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mission score", f"{score}/100")
    col2.metric("Final price", f"${final_price:,.0f}")
    col3.metric("Debt coverage", f"{coverage_final*100:.2f}%")
    col4.metric("Allies joined", f"{int(adopters_final)}")
    col5.metric("Mark-to-market dent", f"{dent_pct*100:.2f}%")
    st.progress(min(1.0, score / 100))

    tradable_supply_latest = float(timeseries.get("effective_float", timeseries.get("tradable_float", 10_000_000.0)).iloc[-1])
    tradable_default_m = max(min(tradable_supply_latest / 1_000_000.0, 19.0), 1.0)

    st.markdown("### How High Could It Go?")
    st.caption("Explore implied price ceilings alongside the model's own percentiles.")
    tam_col, share_col, supply_col = st.columns(3)
    tam_trillions = tam_col.slider("Total Market (TAM)", min_value=5.0, max_value=30.0, value=10.0, step=0.5, help="Pick a big store-of-value market size (like gold).")
    btc_share_pct = share_col.slider("BTC Share", min_value=0.0, max_value=0.5, value=0.10, step=0.01, help="What % of that market might BTC capture?")
    tradable_supply_m = supply_col.slider("Tradable Supply (millions)", min_value=1.0, max_value=19.0, value=round(tradable_default_m, 1), step=0.1, help="Coins that actually trade (not the 21M cap).")

    cofer_col, reserve_col, _ = st.columns(3)
    cofer_trillions = cofer_col.slider("Reserves (COFER)", min_value=5.0, max_value=15.0, value=10.0, step=0.5, help="Central-bank reserves size in USD.")
    reserve_share_pct = reserve_col.slider("Reserve Share", min_value=0.0, max_value=0.20, value=0.05, step=0.01, help="What % of reserves might be in BTC?")

    tam_inputs = TAMInputs(tam_usd=tam_trillions * 1e12, btc_share=btc_share_pct, tradable_supply=tradable_supply_m * 1e6)
    cofer_inputs = COFERInputs(cofer_usd=cofer_trillions * 1e12, reserve_share=reserve_share_pct, tradable_supply=tradable_supply_m * 1e6)

    price_percentiles: Dict[str, float]
    mc_payload = None
    if sim_mode == "Monte Carlo":
        years = max(1, int(round(horizon / 365)))
        mc_config = {
            "years": years,
            "start_price": float(cfg.market.init_price_usd),
            "start_effective_float": float(tradable_supply_latest),
            "postcap_enabled": bool(tuned_cfg.postcap.enabled),
            "postcap_year_index": int(tuned_cfg.postcap.switch_year_index),
            "alpha_postcap_delta": float(tuned_cfg.postcap.alpha_postcap_delta),
            "demand_growth_postcap_delta": float(tuned_cfg.postcap.demand_growth_postcap_delta),
            "hodl_reflexivity": float(tuned_cfg.postcap.hodl_reflexivity),
            "min_tradable_float": float(tuned_cfg.postcap.min_tradable_float),
            "noise_vol": float(tuned_cfg.objective.residual_vol),
        }
        try:
            mc_result = run_mc(mc_config, draws=mc_draws or 32, seed=tuned_cfg.objective.seed)
            mc_payload = _mc_payload(mc_result.percentiles, mc_result.series, mc_result.meta)
            mission_state["mc_artifact"] = mc_payload
            _persist_mc(mc_payload)
            price_percentiles = mc_result.percentiles.get("price", {})
        except Exception as exc:  # pragma: no cover - UI feedback
            st.warning(f"Monte Carlo run failed: {exc}")
            cached = mission_state.get("mc_artifact", {})
            price_percentiles = cached.get("percentiles", {}).get("price", {})
    else:
        price_percentiles = {}

    if not price_percentiles:
        price_percentiles = {"p10": final_price, "p50": final_price, "p90": final_price}

    ceiling_rows = build_ceiling_table(tam_inputs, cofer_inputs, price_percentiles)
    mission_state["tam_inputs"] = {"tam_usd": tam_inputs.tam_usd, "btc_share": tam_inputs.btc_share, "tradable_supply": tam_inputs.tradable_supply}
    mission_state["cofer_inputs"] = {"cofer_usd": cofer_inputs.cofer_usd, "reserve_share": cofer_inputs.reserve_share, "tradable_supply": cofer_inputs.tradable_supply}
    mission_state["percentiles"] = price_percentiles
    if mc_payload:
        mission_state["mc_artifact"] = mc_payload

    focus_price = price_percentiles.get("p50", final_price)
    st.metric("Implied Price", f"${focus_price:,.0f}")
    st.caption(
        f"Model P10 / P50 / P90: ${price_percentiles['p10']:,.0f} / ${price_percentiles['p50']:,.0f} / ${price_percentiles['p90']:,.0f}"
    )

    if ceiling_rows:
        display_df = pd.DataFrame(ceiling_rows).copy()
        for col in ("value", "p10", "p50", "p90"):
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda v: f"${v:,.0f}")
        st.table(display_df)
    else:
        st.info("Provide TAM/COFER inputs to evaluate implied ceilings.")
    st.caption("These implied ceilings are sensitivity checks, not forecasts.")


    if cascade.total_adopters > 0:
        st.markdown("### Adoption cascade")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total adopters", f"{cascade.total_adopters}")
        time_to_half = "n/a" if cascade.time_to_half is None else f"{cascade.time_to_half} days"
        c2.metric("Time to 50%", time_to_half)
        c3.metric("Avg adopters/day", f"{cascade.adoption_rate:.2f}")

    if selected_pb is not None:
        weight = f"{selected_pb.probability:.0%}" if selected_pb.probability is not None else "n/a"
        st.markdown(f"**Playbook:** {selected_pb.name} (weight {weight})")
        if selected_pb.description:
            st.markdown(selected_pb.description)

    if stages_config:
        timeline_df = pd.DataFrame(stages_config).sort_values("start_day")
        timeline_df["countries"] = timeline_df["countries"].apply(lambda codes: ", ".join(codes))
        st.markdown("### Campaign timeline")
        st.table(timeline_df.rename(columns={"start_day": "Day"}))

    yearly = yearly_summary(timeseries)
    if not yearly.empty:
        yearly["mark_to_market"] = yearly["holdings_end"] * yearly["price_end"]
        st.markdown("### Year-by-year outlook")
        st.dataframe(yearly, use_container_width=True, hide_index=True)

    st.subheader("Price trajectory")
    st.line_chart(timeseries.set_index("day")["price"], use_container_width=True)

    st.subheader("Debt coverage over time")
    st.line_chart(timeseries.set_index("day")["debt_coverage"], use_container_width=True)

    st.subheader("Adoption curve")
    st.line_chart(timeseries.set_index("day")["adopters"], use_container_width=True)

    st.markdown("### Network intel")
    centrality = compute_network_centrality(catalog).head(8)
    st.dataframe(centrality, use_container_width=True, hide_index=True)

    st.markdown("### Mission log")
    log_df = pd.DataFrame(engine.adoption_log, columns=["day", "country"])
    if not log_df.empty:
        st.dataframe(log_df.sort_values("day"), use_container_width=True, hide_index=True)
    else:
        st.info("No additional countries joined during this mission. Increase p or q, add more blocs, or extend the mission horizon.")

    st.markdown("### Export")
    st.caption(
        f"Mark-to-market portfolio value: ${mark_to_market:,.0f} (debt dent {dent_pct*100:.2f}% of baseline ${cfg.objective.debt_baseline_usd:,.0f})"
    )
    st.download_button(
        "Download timeseries CSV",
        data=timeseries.to_csv(index=False),
        file_name="bdssim_mission_timeseries.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
