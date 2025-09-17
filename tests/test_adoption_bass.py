import math

import numpy as np
import pytest

from bdssim.adoption.bass import BassState
from bdssim.config import AdoptionBassParams


def _make_state(p: float, q: float, m: int) -> BassState:
    params = AdoptionBassParams(p=p, q=q, m=m)
    return BassState(params=params, candidates=[f"C{i}" for i in range(m)], stochastic=False)


def test_adb01_bounds_monotonicity() -> None:
    """AD-B01: Adoption is monotone, integral, and bounded by m."""
    state = _make_state(0.02, 0.3, 20)
    history = []
    for _ in range(40):
        state.step()
        history.append(state.adopters_count)
    assert all(0 <= val <= 20 for val in history)
    assert all(history[i] <= history[i + 1] for i in range(len(history) - 1))


def test_adb02_no_innovation_no_imitation() -> None:
    """AD-B02: Zero parameters ? no adoption."""
    state = _make_state(0.0, 0.0, 20)
    for _ in range(10):
        state.step()
    assert state.adopters_count == 0


def test_adb03_innovators_only() -> None:
    """AD-B03: Innovator share follows p*(m-A) in aggregate."""
    p = 0.05
    state = _make_state(p, 0.0, 10)
    cumulative_expected = 0.0
    for _ in range(10):
        expected = p * (10 - state.adopters_count)
        cumulative_expected += expected
        state.step()
    assert state.adopters_count == pytest.approx(round(cumulative_expected), rel=0.2)


def test_adb04_s_curve_shape() -> None:
    """AD-B04: Adoption increments accelerate then decelerate (S-curve)."""
    state = _make_state(0.01, 0.30, 20)
    increments = []
    prev = 0
    for _ in range(30):
        state.step()
        increments.append(state.adopters_count - prev)
        prev = state.adopters_count
    peak_idx = int(np.argmax(increments))
    assert peak_idx > 0
    assert increments[peak_idx] >= increments[0]
    assert increments[peak_idx] >= increments[-1]


def test_adb05_peak_time_matches_formula() -> None:
    """AD-B05: Peak time approximates ln(q/p)/(p+q)."""
    params = (0.01, 0.30, 20)
    state = _make_state(*params)
    increments = []
    prev = 0
    for day in range(60):
        state.step()
        increments.append((day, state.adopters_count - prev))
        prev = state.adopters_count
    peak_day = max(increments, key=lambda x: x[1])[0]
    p, q, _ = params
    t_star = math.log(q / p) / (p + q)
    assert peak_day == pytest.approx(t_star, rel=0.25)


def test_adb06_higher_q_faster() -> None:
    """AD-B06: Higher imitation parameter accelerates adoption."""
    base = _make_state(0.01, 0.2, 20)
    fast = _make_state(0.01, 0.4, 20)
    for _ in range(20):
        base.step()
        fast.step()
    assert fast.adopters_count >= base.adopters_count
