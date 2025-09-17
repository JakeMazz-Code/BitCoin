import json

import pytest

from bdssim.data.market_data import fetch_spot_price


class DummyResponse:
    def __init__(self, payload: dict[str, float]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - simple stub
        return

    def json(self) -> dict[str, float]:
        return self._payload


def test_fetch_spot_price_parses(monkeypatch):
    def fake_get(url, params, headers, timeout):  # type: ignore[override]
        assert "bitcoin" in params["ids"]
        assert params["vs_currencies"] == "usd"
        return DummyResponse({"bitcoin": {"usd": 64000}})

    monkeypatch.setattr("bdssim.data.market_data.requests.get", fake_get)
    price = fetch_spot_price()
    assert price == pytest.approx(64000.0)


def test_fetch_spot_price_raises_on_bad_payload(monkeypatch):
    def fake_get(url, params, headers, timeout):  # type: ignore[override]
        return DummyResponse({"unexpected": {"usd": 1}})

    monkeypatch.setattr("bdssim.data.market_data.requests.get", fake_get)
    with pytest.raises(ValueError):
        fetch_spot_price()
