"""Implied price ceiling calculators for TAM and COFER shares."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TAMInputs:
    tam_usd: float
    btc_share: float
    tradable_supply: float

    def __post_init__(self) -> None:
        tam = float(self.tam_usd)
        share = float(self.btc_share)
        supply = float(self.tradable_supply)
        if tam <= 0:
            raise ValueError("tam_usd must be positive")
        if supply <= 0:
            raise ValueError("tradable_supply must be positive")
        if not 0.0 <= share <= 1.0:
            raise ValueError("btc_share must be between 0 and 1")
        object.__setattr__(self, "tam_usd", tam)
        object.__setattr__(self, "btc_share", share)
        object.__setattr__(self, "tradable_supply", supply)


@dataclass(frozen=True)
class COFERInputs:
    cofer_usd: float
    reserve_share: float
    tradable_supply: float

    def __post_init__(self) -> None:
        pool = float(self.cofer_usd)
        share = float(self.reserve_share)
        supply = float(self.tradable_supply)
        if pool <= 0:
            raise ValueError("cofer_usd must be positive")
        if supply <= 0:
            raise ValueError("tradable_supply must be positive")
        if not 0.0 <= share <= 1.0:
            raise ValueError("reserve_share must be between 0 and 1")
        object.__setattr__(self, "cofer_usd", pool)
        object.__setattr__(self, "reserve_share", share)
        object.__setattr__(self, "tradable_supply", supply)


def tam_share_ceiling(inputs: TAMInputs) -> float:
    """Return implied BTC price given a TAM share."""

    return inputs.tam_usd * inputs.btc_share / inputs.tradable_supply


def cofer_share_ceiling(inputs: COFERInputs) -> float:
    """Return implied BTC price given a COFER reserve share."""

    return inputs.cofer_usd * inputs.reserve_share / inputs.tradable_supply


def build_ceiling_table(
    tam: Optional[TAMInputs],
    cofer: Optional[COFERInputs],
    price_percentiles: Dict[str, float],
) -> List[Dict[str, float]]:
    """Construct rows for a ceiling comparison table."""

    rows: List[Dict[str, float]] = []
    p10 = price_percentiles.get("p10")
    p50 = price_percentiles.get("p50")
    p90 = price_percentiles.get("p90")
    if tam is not None:
        rows.append(
            {
                "metric": "TAM share implied price",
                "value": tam_share_ceiling(tam),
                "p10": p10,
                "p50": p50,
                "p90": p90,
            }
        )
    if cofer is not None:
        rows.append(
            {
                "metric": "COFER share implied price",
                "value": cofer_share_ceiling(cofer),
                "p10": p10,
                "p50": p50,
                "p90": p90,
            }
        )
    return rows


__all__ = [
    "TAMInputs",
    "COFERInputs",
    "tam_share_ceiling",
    "cofer_share_ceiling",
    "build_ceiling_table",
]
