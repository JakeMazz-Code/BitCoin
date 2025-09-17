"""Market data helpers (live price fetch)."""

from __future__ import annotations

from typing import Any

import requests

COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
HEADERS = {"Accept": "application/json"}
USER_AGENT = "bdssim/0.1"


def fetch_spot_price(asset: str = "bitcoin", currency: str = "usd") -> float:
    """Fetch current spot price via CoinGecko."""

    params = {"ids": asset, "vs_currencies": currency}
    headers = dict(HEADERS, **{"User-Agent": USER_AGENT})
    response = requests.get(COINGECKO_URL, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    try:
        value = data[asset][currency]
    except KeyError as exc:  # pragma: no cover - depends on API stability
        raise ValueError(f"Unexpected response format: {data}") from exc
    return float(value)


__all__ = ["fetch_spot_price"]
