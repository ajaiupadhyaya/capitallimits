from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fdq.data.provenance import validate_provenance
from fdq.frictions.config import FrictionConfig
from fdq.validation.walkforward import grid_combos, make_folds, walk_forward

FIX = Path(__file__).parent / "fixtures" / "SPY.parquet"


def _real_bars() -> pd.DataFrame:
    validate_provenance(FIX)
    spy = pd.read_parquet(FIX)
    return pd.concat({"SPY": spy}, axis=1)


def test_folds_have_no_leakage() -> None:
    idx = pd.date_range("2016-06-01", "2022-05-31", freq="B")
    folds = make_folds(idx, n_folds=4, scheme="expanding")
    assert len(folds) == 4
    for f in folds:
        assert f.test_start > f.train_end  # test strictly after train


def test_grid_combos_cartesian() -> None:
    combos = grid_combos({"fast": [10, 20], "slow": [50]})
    assert {"fast": 10, "slow": 50} in combos and len(combos) == 2


def test_walk_forward_runs_on_real_fixture() -> None:
    bars = _real_bars()
    if len(bars.index) < 200:
        pytest.skip("fixture too short for walk-forward")
    res = walk_forward(
        "ma_crossover",
        {"symbol": "SPY"},
        {"fast": [10, 20], "slow": [50]},
        bars,
        tier=5.0,
        friction=FrictionConfig(),
        macro=None,
        n_folds=3,
    )
    assert len(res.fold_params) == 3
    assert res.n_trials == res.trial_sharpes.size
    assert res.n_trials >= 3  # >= 1 combo evaluated per fold
    assert isinstance(res.oos_equity, pd.Series)
    # Trial Sharpes must be per-bar (period) units for DSR, not annualized (~15x larger).
    assert abs(res.trial_sharpes).max() < 0.5


def test_in_sample_matrix_shape() -> None:
    from fdq.validation.walkforward import in_sample_return_matrix

    bars = _real_bars()
    start = bars.index.min().date()
    end = bars.index.max().date()
    mat, sr = in_sample_return_matrix(
        "ma_crossover", {"symbol": "SPY"}, {"fast": [10, 20], "slow": [50]},
        bars, 5.0, FrictionConfig(), None, start, end,
    )
    assert mat.shape[1] == 2 and sr.shape[0] == 2
    assert mat.shape[0] > 0
