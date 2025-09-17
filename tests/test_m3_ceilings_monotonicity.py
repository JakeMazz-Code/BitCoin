from bdssim.ceilings import COFERInputs, TAMInputs, cofer_share_ceiling, tam_share_ceiling


def test_tam_share_ceiling_monotonicity() -> None:
    base = TAMInputs(tam_usd=15e12, btc_share=0.10, tradable_supply=10_000_000)
    p0 = tam_share_ceiling(base)
    assert tam_share_ceiling(TAMInputs(tam_usd=20e12, btc_share=0.10, tradable_supply=10_000_000)) > p0
    assert tam_share_ceiling(TAMInputs(tam_usd=15e12, btc_share=0.20, tradable_supply=10_000_000)) > p0
    assert tam_share_ceiling(TAMInputs(tam_usd=15e12, btc_share=0.10, tradable_supply=15_000_000)) < p0


def test_cofer_share_ceiling_monotonicity() -> None:
    base = COFERInputs(cofer_usd=12e12, reserve_share=0.02, tradable_supply=10_000_000)
    p0 = cofer_share_ceiling(base)
    assert cofer_share_ceiling(COFERInputs(cofer_usd=14e12, reserve_share=0.02, tradable_supply=10_000_000)) > p0
    assert cofer_share_ceiling(COFERInputs(cofer_usd=12e12, reserve_share=0.05, tradable_supply=10_000_000)) > p0
    assert cofer_share_ceiling(COFERInputs(cofer_usd=12e12, reserve_share=0.02, tradable_supply=15_000_000)) < p0
