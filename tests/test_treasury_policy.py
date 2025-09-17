from bdssim.config import PolicyParams, SellRuleParams, VenueMix
from bdssim.treasury.policy import Policy


def test_policy_no_sell_when_disabled() -> None:
    params = PolicyParams(
        start_day=0,
        usd_per_day=1_000_000_000.0,
        max_frac_adv=0.05,
        venue_mix=VenueMix(otc=0.7, cex=0.3),
        slippage_budget_bps=30.0,
        sell_rules=SellRuleParams(enabled=False, take_profit_bands=[100_000], tranche_fraction=0.2),
    )
    policy = Policy("Test", params, reference_sigma=0.04)
    assert policy.evaluate_sales(day=100, price=200_000.0, holdings_btc=10_000.0) == 0.0
