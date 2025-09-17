import math

import numpy as np
import pytest

from bdssim.market.impact import (
    ImpactPropagator,
    apply_impact,
    price_update,
    split_impact,
    square_root_impact,
    venue_adjustment,
)


def test_mi_01_square_root_scaling() -> None:
    """MI-01: Square-root scaling and homogeneity."""
    r0 = square_root_impact(0.0, adv_usd=100.0, sigma=0.10, Y=1.0)
    r25 = square_root_impact(25.0, adv_usd=100.0, sigma=0.10, Y=1.0)
    r100 = square_root_impact(100.0, adv_usd=100.0, sigma=0.10, Y=1.0)
    assert r0 == pytest.approx(0.0, abs=1e-12)
    assert r25 == pytest.approx(0.05, abs=1e-12)
    assert r100 == pytest.approx(0.10, abs=1e-12)
    r50 = square_root_impact(50.0, 100.0, 0.10, 1.0)
    assert r50 == pytest.approx(math.sqrt(2.0) * r25, abs=1e-12)


def test_mi_02_liquidity_sensitivity() -> None:
    """MI-02: Impact halves when ADV doubles."""
    r_adv1 = square_root_impact(25.0, adv_usd=100.0, sigma=0.10, Y=1.0)
    r_adv2 = square_root_impact(25.0, adv_usd=200.0, sigma=0.10, Y=1.0)
    assert r_adv2 == pytest.approx(r_adv1 / math.sqrt(2.0), abs=1e-12)


def test_mi_03_buy_sell_asymmetry() -> None:
    """MI-03: Sell-side coefficient scales impact."""
    r_buy = square_root_impact(100.0, 100.0, 0.04, Y=0.70)
    r_sell = square_root_impact(100.0, 100.0, 0.04, Y=0.84)
    assert abs(r_sell) / r_buy == pytest.approx(0.84 / 0.70, abs=1e-12)


def test_mi_04_partition() -> None:
    """MI-04: Permanent vs. temporary partition."""
    permanent, temporary = split_impact(0.05, phi=0.20)
    assert permanent == pytest.approx(0.01, abs=1e-12)
    assert temporary == pytest.approx(0.04, abs=1e-12)
    assert permanent + temporary == pytest.approx(0.05, abs=1e-12)


def test_mi_05_decay() -> None:
    """MI-05: Exponential decay magnitude."""
    propagator = ImpactPropagator(3.0, 1.0)
    propagator.step(0.04)
    residual = 0.0
    for _ in range(3):
        residual = propagator.step(0.0)
    assert residual == pytest.approx(0.04 * math.exp(-1.0), abs=1e-6)
    for _ in range(6):
        residual = propagator.step(0.0)
    assert residual == pytest.approx(0.04 * math.exp(-3.0), abs=1e-6)


def test_mi_06_superposition() -> None:
    """MI-06: Residual impact sums across shocks."""
    propagator = ImpactPropagator(2.0)
    propagator.step(0.03)
    propagator.step(0.02)
    residual = propagator.step(0.0)
    manual = 0.03 * math.exp(-2.0 / 2.0) + 0.02 * math.exp(-1.0 / 2.0)
    assert residual == pytest.approx(manual, abs=1e-12)


def test_mi_07_venue_mix_discount() -> None:
    """MI-07: Venue visibility discount applies to OTC share."""
    raw = 0.05
    adjusted = venue_adjustment(raw, {"otc": 0.7, "cex": 0.3}, 0.6)
    assert adjusted == pytest.approx(0.72 * raw, abs=1e-12)


def test_mi_08_price_update() -> None:
    """MI-08: Golden price update without noise."""
    _, temp = apply_impact(price=100.0, relative_impact=0.05, phi=0.20)
    perm, _ = split_impact(0.05, 0.20)
    price1 = price_update(100.0, perm, temp, noise=0.0)
    assert price1 == pytest.approx(105.0, abs=1e-12)
