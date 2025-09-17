import networkx as nx

from bdssim.adoption.countries import Country, CountryCatalog
from bdssim.adoption.threshold import ThresholdAdoptionModel
from bdssim.config import AdoptionThresholdParams


def _make_catalog() -> CountryCatalog:
    countries = [
        Country("USA", "United States", 23e12, 3.8e11, "G7", 1.0, 0.55),
        Country("CAN", "Canada", 2e12, 9e10, "G7", 0.8, 0.57),
        Country("MEX", "Mexico", 1.3e12, 2e10, "G20", 0.6, 0.6),
        Country("BRA", "Brazil", 1.8e12, 3.5e11, "BRICS", 0.65, 0.58),
    ]
    graph = nx.Graph()
    for country in countries:
        graph.add_node(country.code)
    edges = [
        ("USA", "CAN", 0.9),
        ("USA", "MEX", 0.8),
        ("MEX", "BRA", 0.6),
        ("CAN", "BRA", 0.4),
    ]
    for src, dst, weight in edges:
        graph.add_edge(src, dst, weight=weight)
    return CountryCatalog(countries, graph)


def test_threshold_tipping() -> None:
    catalog = _make_catalog()
    params_high = AdoptionThresholdParams(
        default_theta=0.6,
        theta_std=0.0,
        alpha_momentum=0.1,
        alpha_liquidity=0.1,
        alpha_peer=0.3,
        policy_penalty=0.15,
    )
    params_low = params_high.model_copy(update={"default_theta": 0.55})
    high = ThresholdAdoptionModel(catalog, params_high)
    low = ThresholdAdoptionModel(catalog, params_low)
    high.adopters = ["USA"]
    low.adopters = ["USA"]
    for _ in range(5):
        high.step(price_momentum=0.0, liquidity_score=0.8)
        low.step(price_momentum=0.0, liquidity_score=0.8)
    assert len(low.adopters) >= len(high.adopters)
    assert len(low.adopters) > 1
