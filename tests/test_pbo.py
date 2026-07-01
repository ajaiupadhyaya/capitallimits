from __future__ import annotations

import numpy as np

from fdq.validation.pbo import probability_backtest_overfitting


def test_dominant_config_has_low_pbo() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01, size=(500, 8))
    noise[:, 0] += 0.02  # column 0 dominates in and out of sample
    pbo = probability_backtest_overfitting(noise, n_splits=10)
    assert pbo < 0.3


def test_pure_noise_selection_is_flagged_as_overfit() -> None:
    # No-skill strategies: picking the in-sample winner does not persist OOS.
    # Under CSCV's complementary IS/OOS split the winner mechanically reverts,
    # so PBO trends high — correctly flagging the selection as overfit.
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, size=(500, 8))
    pbo = probability_backtest_overfitting(noise, n_splits=10)
    assert pbo > 0.6


def test_pbo_rejects_single_config() -> None:
    import pytest

    with pytest.raises(ValueError, match="N >= 2"):
        probability_backtest_overfitting(np.zeros((100, 1)))
