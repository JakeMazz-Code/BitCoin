from bdssim.accounting.ledger import Ledger


def test_fifo_cost_basis_and_coverage() -> None:
    ledger = Ledger()
    ledger.buy_usd(100_000_000, 50_000)
    ledger.buy_usd(50_000_000, 60_000)
    avg_cost = ledger.average_cost()
    assert 50_000 <= avg_cost <= 55_000
    proceeds, pnl = ledger.sell_btc(1000, 70_000)
    assert proceeds > 0
    assert pnl > 0
    coverage = ledger.debt_coverage(37_450_000_000_000)
    assert coverage >= 0
