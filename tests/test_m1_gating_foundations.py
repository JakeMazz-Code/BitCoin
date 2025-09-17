from bdssim.engine.adoption import unlocked_for_adoption


def test_unlocked_for_adoption_requires_both_conditions() -> None:
    assert unlocked_for_adoption(t=1, us_progress=0.30, min_lag_years=1, progress_threshold=0.25) is True
    assert unlocked_for_adoption(t=0, us_progress=0.30, min_lag_years=1, progress_threshold=0.25) is False
    assert unlocked_for_adoption(t=2, us_progress=0.10, min_lag_years=1, progress_threshold=0.25) is False
