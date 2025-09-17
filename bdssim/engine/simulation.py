"""Simulation engine orchestrating modules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

BASE_CONFIG_PATH = Path("configs/base.yaml")

from bdssim.accounting.ledger import Ledger
from bdssim.adoption.bass import BassState
from bdssim.adoption.countries import CountryCatalog, load_countries, load_country_graph
from bdssim.adoption.threshold import ThresholdAdoptionModel
from bdssim.config import Config, load_config
from bdssim.engine.state import MarketState
from bdssim.market.exogenous import ExogenousManager, build_exogenous_manager
from bdssim.market.impact import ImpactPropagator, apply_impact, square_root_impact
from bdssim.supply.issuance import circulating_supply, effective_float, issuance_for_day, miner_sell
from bdssim.treasury.policy import CountryPolicy, Policy, PolicyDecision, USPolicy
from bdssim.utils.io import save_table, write_config_snapshot
from bdssim.utils.rng import RNGManager


@dataclass
class PlayStage:
    name: str
    start_day: int
    countries: List[str]


@dataclass
class SimulationResults:
    timeseries: pd.DataFrame
    adoption_log: List[Tuple[int, str]]


class SimulationEngine:
    """Daily simulation applying policy execution, adoption, and supply."""

    US_CODE = "USA"

    def __init__(
        self,
        config: Config,
        data_root: str | Path | None = None,
        initial_adopters: Iterable[str] | None = None,
        playbook: Iterable[dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.data_root = Path(data_root or Path.cwd())
        self.rng = RNGManager(config.objective.seed)
        self.market_state = MarketState(
            day=0,
            price=config.market.init_price_usd,
            sigma=config.market.init_sigma,
            adv_usd=config.market.adv_usd,
            depth_score=config.market.depth_score,
            liquidity_score=config.market.depth_score,
        )
        self.market_state.update_liquidity(config.market.adv_usd, config.market.depth_score)
        self.propagator = ImpactPropagator(config.impact.temp_decay_tau_days, config.impact.temp_decay_lambda)
        self.ledger_us = Ledger()
        self.country_ledgers: Dict[str, Ledger] = {}
        self.us_policy = USPolicy("US", config.policy_us, config.market.init_sigma)
        self.us_policy.scheduler.Y = config.impact.Y_buy
        self.country_policies: Dict[str, CountryPolicy] = {}
        self.adopted_countries: set[str] = set()
        self._pending_stage_adopters: list[tuple[str, int]] = []
        self._min_lag_days = int(round(config.adoption.min_lag_years * 365))
        self._progress_threshold = float(config.adoption.progress_threshold)
        plan_total = config.policy_us.usd_per_day * config.objective.horizon_days
        self._us_plan_total_usd = plan_total if plan_total > 0 else 1.0
        self._us_cumulative_spend = 0.0
        self.postcap = config.postcap
        self._postcap_enabled = self.postcap.enabled
        self._postcap_switch_day = self.postcap.switch_year_index * 365
        if self._postcap_switch_day >= config.objective.horizon_days:
            self._postcap_enabled = False
        self._base_alpha = float(config.market.scarcity_alpha)
        self._base_demand_growth = float(config.market.demand_growth)
        self._min_tradable_float = float(self.postcap.min_tradable_float)
        base_supply = effective_float(circulating_supply(0), config.supply.lost_btc)
        base_effective = max(base_supply, self._min_tradable_float)
        self._base_float = base_effective
        self._effective_float = base_effective
        self._float_base_prev = base_supply if base_supply > 0 else base_effective
        self._demand_index = 1.0
        self._year_length_days = 365
        self._year_start_price = float(config.market.init_price_usd)
        self._prior_year_return = 0.0
        self._last_reflexivity_delta = 0.0
        self.exogenous = self._build_exogenous_manager()
        self.catalog = self._build_catalog()
        adoption_rng = self.rng.generator("adoption")
        if config.adoption.model == "bass":
            candidates = self.catalog.codes()
            self.adoption_model = BassState(config.adoption.bass, candidates, adoption_rng)
        else:
            self.adoption_model = ThresholdAdoptionModel(self.catalog, config.adoption.threshold, adoption_rng)
        self.adoption_log: List[Tuple[int, str]] = []
        if initial_adopters:
            self._seed_initial_adopters(initial_adopters)
        self.playbook_schedule: List[PlayStage] = self._build_playbook(playbook)
        self._playbook_index = 0
        self.residual_rng = self.rng.generator("returns")

    def _build_playbook(self, playbook: Iterable[dict[str, Any]] | None) -> List[PlayStage]:
        stages: List[PlayStage] = []
        source = playbook
        if source is None and self.config.playbook:
            source = [stage.model_dump() for stage in self.config.playbook]
        if source:
            for stage in source:
                name = stage.get("name", "Stage")
                start_day = int(stage.get("start_day", 0))
                countries = list(stage.get("countries", []))
                if countries:
                    stages.append(PlayStage(name=name, start_day=start_day, countries=countries))
        stages.sort(key=lambda s: s.start_day)
        return stages

    def _seed_initial_adopters(self, seeds: Iterable[str]) -> None:
        for code in seeds:
            if code not in self.catalog.codes():
                continue
            self._register_country_policy(code, day=0)
            self._mark_adopted(code, -1)

    def _build_catalog(self) -> CountryCatalog:
        countries_path = self.data_root / self.config.countries.country_csv
        graph_path = self.data_root / self.config.countries.graph_csv
        countries = load_countries(countries_path)
        graph = load_country_graph(graph_path, [c.code for c in countries])
        return CountryCatalog(countries, graph)

    def _build_exogenous_manager(self) -> ExogenousManager:
        etf = self.config.exogenous.etf_flow_csv
        auctions = self.config.exogenous.auctions_csv
        etf_path = str(self.data_root / etf) if etf else None
        auction_path = str(self.data_root / auctions) if auctions else None
        return build_exogenous_manager(etf_path, auction_path)

    def _register_country_policy(self, code: str, day: int, adopt_immediately: bool = False) -> None:
        if code in self.country_policies:
            return
        if code not in self.catalog.codes():
            return
        country = self.catalog.country(code)
        total_budget = country.reserves_usd * self.config.countries.reserves_fraction_on_adopt
        remaining_days = max(self.config.objective.horizon_days - day, 1)
        usd_per_day = total_budget / remaining_days
        params = self.config.policy_us.model_copy(update={
            "start_day": day,
            "usd_per_day": usd_per_day,
            "sell_rules": self.config.policy_us.sell_rules.model_copy(),
        })
        policy = CountryPolicy(code, params, self.config.market.init_sigma)
        policy.scheduler.Y = self.config.impact.Y_buy
        self.country_policies[code] = policy
        self.country_ledgers[code] = Ledger()
        if adopt_immediately:
            self._mark_adopted(code, day)

    def _mark_adopted(self, code: str, day: int) -> None:
        if code in self.adopted_countries:
            return
        self.adopted_countries.add(code)
        self._pending_stage_adopters = [(pending_code, trigger) for pending_code, trigger in self._pending_stage_adopters if pending_code != code]
        if isinstance(self.adoption_model, BassState):
            if code not in self.adoption_model.adopters:
                self.adoption_model.adopters.append(code)
        else:
            if code not in self.adoption_model.adopters:
                self.adoption_model.adopters.append(code)
        self.adoption_log.append((day, code))

    def _adoption_unlocked(self, day: int) -> bool:
        if day < self._min_lag_days:
            return False
        if self._progress_threshold <= 0:
            return True
        return self.us_progress >= self._progress_threshold

    def _allowed_adoption_codes(self, day: int) -> set[str]:
        if self._adoption_unlocked(day):
            return set(self.catalog.codes())
        return {self.US_CODE}

    def _release_pending_stage_adopters(self, day: int) -> None:
        if not self._pending_stage_adopters:
            return
        if not self._adoption_unlocked(day):
            return
        remaining: list[tuple[str, int]] = []
        for code, trigger_day in self._pending_stage_adopters:
            if code in self.adopted_countries:
                continue
            if day >= trigger_day:
                self._mark_adopted(code, day)
            else:
                remaining.append((code, trigger_day))
        self._pending_stage_adopters = remaining

    @property
    def us_progress(self) -> float:
        return min(self._us_cumulative_spend / self._us_plan_total_usd, 1.0)

    def _apply_policy_decision(self, policy: Policy, ledger: Ledger, decision: PolicyDecision, price: float) -> Tuple[float, float]:
        net_usd = decision.net_usd
        visible = decision.visible_usd
        if net_usd > 0:
            ledger.buy_usd(net_usd, price)
        elif net_usd < 0:
            sell_btc = abs(net_usd) / price
            ledger.sell_btc(sell_btc, price)
        if policy is self.us_policy and net_usd > 0:
            self._us_cumulative_spend += net_usd
        return net_usd, visible

    def _stochastic_flow(self) -> float:
        bps = self.config.exogenous.stochastic_flow_bps
        if bps <= 0:
            return 0.0
        adv = self.market_state.adv_usd
        noise = self.residual_rng.normal(0.0, bps / 10_000)
        return float(noise * adv)

    def _trigger_stages(self, day: int) -> None:
        while self._playbook_index < len(self.playbook_schedule) and self.playbook_schedule[self._playbook_index].start_day == day:
            stage = self.playbook_schedule[self._playbook_index]
            for code in stage.countries:
                adopt_now = (code == self.US_CODE) or self._adoption_unlocked(day)
                self._register_country_policy(code, day, adopt_immediately=adopt_now)
                if adopt_now:
                    continue
                if code == self.US_CODE:
                    continue
                if all(existing != code for existing, _ in self._pending_stage_adopters):
                    self._pending_stage_adopters.append((code, day))
            self._playbook_index += 1

    def _build_catalog(self) -> CountryCatalog:
        countries_path = self.data_root / self.config.countries.country_csv
        graph_path = self.data_root / self.config.countries.graph_csv
        countries = load_countries(countries_path)
        graph = load_country_graph(graph_path, [c.code for c in countries])
        return CountryCatalog(countries, graph)

    def _update_adoption(self, day: int) -> List[str]:
        self._release_pending_stage_adopters(day)
        allowed = self._allowed_adoption_codes(day)
        if isinstance(self.adoption_model, BassState):
            new = self.adoption_model.step(allowed=allowed)
        else:
            momentum = self.market_state.momentum
            liquidity = self.market_state.liquidity_score
            new = self.adoption_model.step(momentum, liquidity, allowed=allowed)
        for code in new:
            self._register_country_policy(code, day + 1)
            self._mark_adopted(code, day)
        return new

    def run(self, out_dir: str | Path | None = None) -> SimulationResults:
        records: List[Dict[str, float]] = []
        cfg = self.config
        debt_baseline = cfg.objective.debt_baseline_usd
        horizon = cfg.objective.horizon_days

        for day in range(horizon):
            self._trigger_stages(day)
            self.market_state.day = day
            prev_price = self.market_state.price
            year_index = day // self._year_length_days
            start_of_year = day % self._year_length_days == 0
            reflexivity_delta = 0.0
            if start_of_year and day > 0:
                if self._year_start_price > 0:
                    self._prior_year_return = prev_price / self._year_start_price - 1.0
                else:
                    self._prior_year_return = 0.0
            prior_year_return = self._prior_year_return
            if start_of_year:
                self._year_start_price = prev_price
        
            float_supply = effective_float(circulating_supply(day), cfg.supply.lost_btc)
            float_base_today = max(float_supply, self._min_tradable_float)
            if day == 0:
                self._float_base_prev = float_base_today
                self._effective_float = min(self._effective_float, float_base_today)
            else:
                added = float_base_today - self._float_base_prev
                self._float_base_prev = float_base_today
                if added > 0:
                    self._effective_float = min(self._effective_float + added, float_base_today)
                else:
                    self._effective_float = min(self._effective_float, float_base_today)
            self._effective_float = max(self._effective_float, self._min_tradable_float)
        
            if (
                start_of_year
                and day > 0
                and self._postcap_enabled
                and year_index >= self.postcap.switch_year_index
                and self.postcap.hodl_reflexivity > 0
                and prior_year_return > 0
            ):
                proposed_reduction = self._effective_float * self.postcap.hodl_reflexivity * prior_year_return
                new_effective_float = max(self._effective_float - proposed_reduction, self._min_tradable_float)
                reflexivity_delta = self._effective_float - new_effective_float
                if reflexivity_delta > 0:
                    self._effective_float = new_effective_float
            self._last_reflexivity_delta = reflexivity_delta
        
            postcap_active = self._postcap_enabled and day >= self._postcap_switch_day
            alpha_eff = self._base_alpha + (self.postcap.alpha_postcap_delta if postcap_active else 0.0)
            demand_growth_eff = self._base_demand_growth + (self.postcap.demand_growth_postcap_delta if postcap_active else 0.0)
            float_ratio = self._effective_float / self._base_float if self._base_float > 0 else 1.0
            float_ratio = max(float_ratio, 1e-6)
            scarcity_scale_for_impact = float_ratio ** alpha_eff if alpha_eff else 1.0
            daily_growth = 0.0
            if demand_growth_eff:
                if demand_growth_eff <= -0.999:
                    daily_growth = -0.999 / 365.0
                else:
                    daily_growth = (1.0 + demand_growth_eff) ** (1.0 / 365.0) - 1.0
        
            total_net = 0.0
            total_visible = 0.0
            # US policy execution
            decision_us = self.us_policy.decision(
                day,
                self.market_state.price,
                self.market_state.sigma,
                self.market_state.adv_usd,
                self.market_state.depth_score,
                self.ledger_us.holdings_btc,
            )
            delta_us, visible_us = self._apply_policy_decision(self.us_policy, self.ledger_us, decision_us, self.market_state.price)
            total_net += delta_us
            total_visible += visible_us
            # Country policies
            for code, policy in list(self.country_policies.items()):
                ledger = self.country_ledgers[code]
                decision = policy.decision(
                    day,
                    self.market_state.price,
                    self.market_state.sigma,
                    self.market_state.adv_usd,
                    self.market_state.depth_score,
                    ledger.holdings_btc,
                )
                delta, visible = self._apply_policy_decision(policy, ledger, decision, self.market_state.price)
                total_net += delta
                total_visible += visible
            # Exogenous flows
            exo = self.exogenous.flow_for_day(day)
            total_net += exo
            total_visible += exo
            total_net += self._stochastic_flow()
            # Miner sells from issuance
            issued_btc = issuance_for_day(day)
            miner_usd = miner_sell(issued_btc, cfg.supply.miner_sell_frac) * self.market_state.price
            total_net -= miner_usd
            total_visible -= miner_usd
            # Impact calculation
            visible_order = total_visible
            if visible_order >= 0:
                Y = cfg.impact.Y_buy
                phi = cfg.impact.permanent_phi_buy
            else:
                Y = cfg.impact.Y_sell
                phi = cfg.impact.permanent_phi_sell
            adv = max(self.market_state.adv_usd * max(scarcity_scale_for_impact, 1e-6), 1.0)
            sigma = max(self.market_state.sigma, 1e-6)
            impact_r = square_root_impact(abs(visible_order), adv, sigma, Y)
            impact_r = math.copysign(impact_r, visible_order)
            new_price, temp_component = apply_impact(self.market_state.price, impact_r, phi)
            temp_effect = self.propagator.step(temp_component)
            impacted_price = new_price * (1 + temp_effect)
            noise = float(self.residual_rng.normal(0.0, cfg.objective.residual_vol))
            impacted_price *= math.exp(noise - 0.5 * cfg.objective.residual_vol**2)
            if daily_growth:
                growth_multiplier = 1.0 + daily_growth
                drift_reference = prev_price * growth_multiplier
                impacted_price *= growth_multiplier
                self._demand_index *= growth_multiplier
                if daily_growth > 0:
                    impacted_price = max(impacted_price, drift_reference)
                else:
                    impacted_price = min(impacted_price, drift_reference)
            impacted_price = max(impacted_price, 1.0)
            self.market_state.update_price(impacted_price)
            if self.market_state.price > 0:
                net_btc = total_net / self.market_state.price
                if net_btc > 0:
                    self._effective_float = max(self._effective_float - net_btc, self._min_tradable_float)
                elif net_btc < 0:
                    self._effective_float = min(self._effective_float - net_btc, float_base_today)
            float_ratio_post = self._effective_float / self._base_float if self._base_float > 0 else 1.0
            float_ratio_post = max(float_ratio_post, 1e-6)
            scarcity_scaler = float_ratio_post ** alpha_eff if alpha_eff else 1.0
        
            self.market_state.total_order_usd = total_net
            self.market_state.visible_order_usd = total_visible
            self.market_state.adopters = len(self.adopted_countries)
            # Simple liquidity feedback
            self.market_state.update_liquidity(
                adv_usd=max(cfg.market.adv_usd + 0.1 * total_visible, 1.0),
                depth_score=max(min(cfg.market.depth_score + 0.01 * total_visible / cfg.market.adv_usd, 1.0), 0.0),
            )
            coverage = self.ledger_us.debt_coverage(debt_baseline)
            record = {
                "day": day,
                "price": self.market_state.price,
                "total_order_usd": total_net,
                "visible_order_usd": total_visible,
                "us_holdings_btc": self.ledger_us.holdings_btc,
                "us_avg_cost_usd": self.ledger_us.average_cost(),
                "us_unrealized_pnl_usd": self.ledger_us.unrealized_pnl(self.market_state.price),
                "debt_coverage": coverage,
                "adopters": self.market_state.adopters,
                "effective_float": self._effective_float,
                "circulating_supply": float_supply,
                "postcap_active": 1 if postcap_active else 0,
                "alpha_eff": alpha_eff,
                "demand_growth_eff": demand_growth_eff,
                "demand_index": self._demand_index,
                "scarcity_scaler": scarcity_scaler,
                "prior_year_return": prior_year_return,
                "reflexivity_delta": reflexivity_delta,
            }
            records.append(record)
            self._update_adoption(day)
        timeseries = pd.DataFrame(records)
        if out_dir is not None:
            out = Path(out_dir)
            save_table(timeseries, out, "timeseries")
            summary = pd.DataFrame(
                {
                    "metric": ["final_price", "debt_coverage", "us_holdings_btc"],
                    "value": [
                        timeseries["price"].iloc[-1],
                        timeseries["debt_coverage"].iloc[-1],
                        timeseries["us_holdings_btc"].iloc[-1],
                    ],
                }
            )
            save_table(summary, out, "summary")
            write_config_snapshot(self.config, out)
        return SimulationResults(timeseries=timeseries, adoption_log=self.adoption_log)


def run_years(*, config: dict) -> pd.DataFrame:
    """Run a simplified yearly simulation for testing helpers."""

    cfg_dict = dict(config) if config is not None else {}
    years = int(cfg_dict.get("years", 1))
    years = max(years, 1)
    horizon_days = years * 365

    overrides: dict[str, dict[str, float | int | bool]] = {
        "objective": {"horizon_days": horizon_days}
    }

    if "noise_vol" in cfg_dict:
        overrides["objective"]["residual_vol"] = float(cfg_dict["noise_vol"])

    market_update: dict[str, float] = {}
    if "start_price" in cfg_dict:
        market_update["init_price_usd"] = float(cfg_dict["start_price"])
    if "scarcity_alpha" in cfg_dict:
        market_update["scarcity_alpha"] = float(cfg_dict["scarcity_alpha"])
    if "demand_growth" in cfg_dict:
        market_update["demand_growth"] = float(cfg_dict["demand_growth"])
    if market_update:
        overrides["market"] = market_update

    postcap_update: dict[str, float | int | bool] = {}
    if "postcap_enabled" in cfg_dict:
        postcap_update["enabled"] = bool(cfg_dict["postcap_enabled"])
    if "postcap_year_index" in cfg_dict:
        postcap_update["switch_year_index"] = int(cfg_dict["postcap_year_index"])
    if "alpha_postcap_delta" in cfg_dict:
        postcap_update["alpha_postcap_delta"] = float(cfg_dict["alpha_postcap_delta"])
    if "demand_growth_postcap_delta" in cfg_dict:
        postcap_update["demand_growth_postcap_delta"] = float(cfg_dict["demand_growth_postcap_delta"])
    if "hodl_reflexivity" in cfg_dict:
        postcap_update["hodl_reflexivity"] = float(cfg_dict["hodl_reflexivity"])
    if "min_tradable_float" in cfg_dict:
        postcap_update["min_tradable_float"] = float(cfg_dict["min_tradable_float"])
    if postcap_update:
        overrides["postcap"] = postcap_update

    supply_update: dict[str, float] = {}
    start_float = None
    if "start_effective_float" in cfg_dict:
        start_float = float(cfg_dict["start_effective_float"])
        base_supply = circulating_supply(0)
        lost_btc = max(base_supply - start_float, 0.0)
        supply_update["lost_btc"] = lost_btc
    if supply_update:
        overrides["supply"] = supply_update

    cfg = load_config(BASE_CONFIG_PATH, overrides=overrides)

    engine = SimulationEngine(cfg, data_root=Path('.'), initial_adopters=[SimulationEngine.US_CODE])

    if start_float is not None:
        engine._base_float = start_float
        engine._effective_float = start_float
        engine._float_base_prev = start_float

    result = engine.run()
    ts = result.timeseries.copy()
    year_length = engine._year_length_days
    ts["year"] = (ts["day"] // year_length).astype(int)

    yearly = (
        ts.groupby("year", as_index=False)
        .agg(
            price=("price", "last"),
            tradable_float=("effective_float", "last"),
            circ_supply=("circulating_supply", "last"),
            prior_year_return=("prior_year_return", "last"),
            reflexivity_delta=("reflexivity_delta", "last"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    return yearly



__all__ = ["SimulationEngine", "SimulationResults", "PlayStage", "run_years"]
