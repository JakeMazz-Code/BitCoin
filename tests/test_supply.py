from bdssim.supply.issuance import circulating_supply, effective_float, issuance_for_day


def test_supply_caps() -> None:
    today = circulating_supply(0)
    future = circulating_supply(3650)
    assert today <= 21_000_000
    assert future <= 21_000_000
    issuance = issuance_for_day(1)
    assert issuance >= 0


def test_effective_float() -> None:
    supply = 19_500_000
    float_supply = effective_float(supply, 3_000_000)
    assert float_supply <= supply
    assert float_supply == supply - 3_000_000
