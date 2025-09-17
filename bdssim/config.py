"""Configuration models and loaders for bdssim."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, RootModel, ValidationError, model_validator


class MetaParams(BaseModel):
    name: str = "base"
    description: str | None = None
    version: str | None = None


class MarketParams(BaseModel):
    """Market environment parameters."""

    init_price_usd: float = Field(..., gt=0, description="Initial BTC/USD spot price")
    init_sigma: float = Field(..., gt=0, description="Initial daily volatility (fraction)")
    adv_usd: float = Field(..., gt=0, description="Average daily traded volume (USD)")
    depth_score: float = Field(0.6, ge=0, le=1, description="Relative liquidity depth score")
    spread_bps: float = Field(5.0, ge=0, description="Reference bid/ask spread in basis points")
    scarcity_alpha: float = Field(0.0, ge=0.0, description="Base scarcity curvature exponent")
    demand_growth: float = Field(0.0, ge=-0.99, description="Baseline annual demand drift")


class ImpactParams(BaseModel):
    """Square-root impact and decay parameters."""

    Y_buy: float = Field(0.7, ge=0.1, le=2.0)
    Y_sell: float = Field(0.8, ge=0.1, le=2.0)
    permanent_phi_buy: float = Field(0.25, ge=0, le=1)
    permanent_phi_sell: float = Field(0.30, ge=0, le=1)
    temp_decay_tau_days: float = Field(3.0, gt=0)
    temp_decay_lambda: float = Field(1.0, gt=0)
    otc_visible_fraction: float = Field(0.4, ge=0, le=1, description="Share of OTC volume that becomes visible")


class ExogenousParams(BaseModel):
    """External order flow settings."""

    etf_flow_csv: Optional[str] = Field(None, description="CSV with columns [day, usd_flow]")
    auctions_csv: Optional[str] = None
    stochastic_flow_bps: float = Field(0.0, ge=0, description="Randomized daily flow noise in bps of ADV")


class AdoptionBassParams(BaseModel):
    """Parameters for Bass diffusion model."""

    p: float = Field(..., ge=0, le=1, description="Innovation coefficient")
    q: float = Field(..., ge=0, le=2, description="Imitation coefficient")
    m: int = Field(..., ge=1, description="Market potential (sovereign adopters)")


class PolicyFrictionParams(BaseModel):
    """Policy friction relief schedule keyed by adoption share thresholds."""

    relief_steps: Dict[float, float] = Field(
        default_factory=lambda: {0.25: 0.05, 0.5: 0.05},
        description="Mapping of adoption share thresholds to penalty reductions",
    )

    def reduction(self, share: float) -> float:
        total = 0.0
        for threshold, delta in sorted(self.relief_steps.items()):
            if share >= threshold:
                total += delta
        return total


class AdoptionThresholdParams(BaseModel):
    """Parameters for network threshold contagion."""

    default_theta: float = Field(0.55, ge=0, le=1)
    theta_std: float = Field(0.08, ge=0, le=0.5)
    alpha_momentum: float = Field(0.15, ge=0)
    alpha_liquidity: float = Field(0.10, ge=0)
    alpha_peer: float = Field(0.25, ge=0)
    policy_penalty: float = Field(0.15, ge=0)
    bloc_relief: PolicyFrictionParams = Field(default_factory=PolicyFrictionParams)


class AdoptionParams(BaseModel):
    """Adoption model selection and parameters."""

    model: Literal["bass", "threshold"] = "bass"
    bass: AdoptionBassParams = Field(default_factory=lambda: AdoptionBassParams(p=0.01, q=0.30, m=20))
    threshold: AdoptionThresholdParams = Field(default_factory=AdoptionThresholdParams)
    min_lag_years: float = Field(1.0, ge=0.0, description="Minimum years of U.S. leadership before others can adopt")
    progress_threshold: float = Field(0.25, ge=0.0, le=1.0, description="Share of the U.S. plan completed before others unlock")



class PostCapParams(BaseModel):
    """Parameters controlling the post-cap monetization regime."""

    enabled: bool = True
    switch_year_index: int = Field(3, ge=0, description="Year index (0=first year) when post-cap regime activates")
    alpha_postcap_delta: float = Field(0.10, ge=0.0, description="Incremental scarcity curvature once post-cap is active")
    demand_growth_postcap_delta: float = Field(0.02, ge=-0.99, description="Incremental annual demand drift once post-cap is active")
    hodl_reflexivity: float = Field(0.10, ge=0.0, description="Fraction of positive returns parked as long-term holds")
    min_tradable_float: float = Field(1_000_000.0, ge=0.0, description="Lower bound on tradable float to keep markets liquid")


class SupplyParams(BaseModel):
    """Bitcoin issuance inputs."""

    lost_btc: float = Field(3_000_000, ge=0)
    miner_sell_frac: float = Field(0.35, ge=0, le=1)


class SellRuleParams(BaseModel):
    """Configuration for discretionary sales."""

    enabled: bool = Field(True, description="If False, no discretionary sales occur")
    take_profit_bands: Iterable[float] = Field(default_factory=list)
    tranche_fraction: float = Field(0.2, ge=0, le=1)
    time_lock_days: int = Field(30, ge=0)



class VenueMix(BaseModel):
    """Execution venue mix."""

    otc: float = Field(0.7, ge=0, le=1)
    cex: float = Field(0.3, ge=0, le=1)

    @model_validator(mode="after")
    def validate_sum(cls, values: "VenueMix") -> "VenueMix":
        total = values.otc + values.cex
        if not 0.99 <= total <= 1.01:
            raise ValueError("Venue weights must sum to 1 within tolerance")
        return values


class PolicyParams(BaseModel):
    """General policy schedule inputs."""

    start_day: int = Field(0, ge=0)
    usd_per_day: float = Field(..., ge=0)
    max_frac_adv: float = Field(0.05, ge=0, le=1)
    venue_mix: VenueMix = Field(default_factory=VenueMix)
    slippage_budget_bps: float = Field(30.0, ge=0)
    smart_pacing: bool = Field(True)
    sell_rules: SellRuleParams = Field(default_factory=SellRuleParams)


class CountriesParams(BaseModel):
    """Country data sources and behavioural inputs."""

    country_csv: str
    graph_csv: str
    reserves_fraction_on_adopt: float = Field(0.05, ge=0, le=1)


class ObjectiveParams(BaseModel):
    """Objective function configuration."""

    debt_baseline_usd: float = Field(37_450_000_000_000, ge=0)
    horizon_days: int = Field(1825, ge=1)
    seed: int = Field(42, ge=0)
    residual_vol: float = Field(0.02, ge=0)


class PlaybookStage(BaseModel):
    name: str
    start_day: int = Field(..., ge=0)
    countries: List[str]


class Config(BaseModel):
    """Top-level configuration model."""

    meta: MetaParams = Field(default_factory=MetaParams)
    market: MarketParams
    impact: ImpactParams = Field(default_factory=ImpactParams)
    exogenous: ExogenousParams = Field(default_factory=ExogenousParams)
    adoption: AdoptionParams = Field(default_factory=AdoptionParams)
    supply: SupplyParams = Field(default_factory=SupplyParams)
    policy_us: PolicyParams
    countries: CountriesParams
    objective: ObjectiveParams = Field(default_factory=ObjectiveParams)
    postcap: PostCapParams = Field(default_factory=PostCapParams)
    playbook: List[PlaybookStage] = Field(default_factory=list)


class ConfigRoot(RootModel[Config]):
    """Wrapper to allow Config in root of YAML."""

    root: Config


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def default_config_dict() -> Dict[str, Any]:
    """Return the default configuration as a dictionary."""

    default = Config(
        market=MarketParams(init_price_usd=60_000, init_sigma=0.04, adv_usd=15_000_000_000, depth_score=0.6),
        policy_us=PolicyParams(usd_per_day=1_000_000_000),
        countries=CountriesParams(
            country_csv="data/countries/country_list.csv",
            graph_csv="data/countries/country_graph.csv",
        ),
    )
    return default.model_dump()


def load_config(path: str | Path | None = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load configuration from YAML and merge with defaults."""

    base_dict = default_config_dict()
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            user_data = yaml.safe_load(handle) or {}
        if not isinstance(user_data, dict):
            raise ValidationError(["Config YAML must map to an object"], Config)
        base_dict = _deep_update(base_dict, user_data)
    if overrides:
        base_dict = _deep_update(base_dict, overrides)
    config = Config.model_validate(base_dict)
    return config


__all__ = [
    "Config",
    "MetaParams",
    "MarketParams",
    "ImpactParams",
    "ExogenousParams",
    "AdoptionParams",
    "AdoptionBassParams",
    "AdoptionThresholdParams",
    "SupplyParams",
    "PostCapParams",
    "PolicyParams",
    "CountriesParams",
    "ObjectiveParams",
    "PlaybookStage",
    "load_config",
]

