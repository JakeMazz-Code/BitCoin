"""Bitcoin issuance and float calculations."""

from __future__ import annotations

from dataclasses import dataclass

from bdssim.data.constants import (
    BLOCKS_PER_DAY,
    HALVING_INTERVAL_BLOCKS,
    INITIAL_BLOCK_REWARD,
    BTC_SUPPLY_CAP,
)

BASE_HEIGHT = 840_000


def cumulative_issuance(block_height: int) -> float:
    """Total BTC issued up to ``block_height`` (inclusive)."""

    reward = INITIAL_BLOCK_REWARD
    issued = 0.0
    remaining = block_height
    while remaining > 0 and reward > 0:
        blocks = min(HALVING_INTERVAL_BLOCKS, remaining)
        issued += blocks * reward
        remaining -= blocks
        reward /= 2
    return min(issued, BTC_SUPPLY_CAP)


def circulating_supply(day: int) -> float:
    """Approximate circulating supply ``day`` days from the base height."""

    height = BASE_HEIGHT + max(day, 0) * BLOCKS_PER_DAY
    return min(cumulative_issuance(height), BTC_SUPPLY_CAP)


def miner_sell(new_issuance_btc: float, miner_sell_frac: float) -> float:
    """Return miner sell pressure given new issuance and sell fraction."""

    return new_issuance_btc * miner_sell_frac


def effective_float(total_supply: float, lost_btc: float) -> float:
    """Compute circulating float after lost coins haircut."""

    return max(total_supply - lost_btc, 0.0)


def issuance_for_day(day: int) -> float:
    """New coins issued on ``day``."""

    total_today = circulating_supply(day)
    total_prev = circulating_supply(max(day - 1, 0))
    return max(total_today - total_prev, 0.0)


__all__ = [
    "circulating_supply",
    "miner_sell",
    "effective_float",
    "issuance_for_day",
]
