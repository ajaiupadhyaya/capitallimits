from __future__ import annotations

import pytest

from fdq.validation.power import mds_over_horizon, minimum_detectable_sharpe


def test_mds_one_year_daily_equals_z_sum() -> None:
    # For n == periods_per_year the sqrt(periods/n) factor is 1, so the annualized
    # MDS equals z_{1-alpha} + z_{power} ~= 1.645 + 0.842 = 2.486.
    mds = minimum_detectable_sharpe(252, periods_per_year=252, alpha=0.05, power=0.80)
    assert mds == pytest.approx(2.486, abs=0.01)


def test_mds_decreases_with_more_data() -> None:
    assert minimum_detectable_sharpe(504) < minimum_detectable_sharpe(252)


def test_mds_over_horizon_months() -> None:
    one_year = mds_over_horizon(12)
    two_year = mds_over_horizon(24)
    assert one_year == pytest.approx(2.486, abs=0.02)
    assert two_year < one_year
