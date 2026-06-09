"""Validation module tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fdq.validation.bootstrap import bootstrap_ruin_analysis
from fdq.validation.dsr import deflated_sharpe, probabilistic_sharpe
from fdq.validation.metrics import max_drawdown, sharpe
from fdq.validation.pbo import probability_backtest_overfitting


def test_sharpe_positive_on_uptrend() -> None:
    returns = pd.Series(np.linspace(0.0005, 0.0015, 100))
    assert sharpe(returns) > 0


def test_psr_on_positive_returns() -> None:
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.001, 0.01, 252))
    psr = probabilistic_sharpe(returns, sr_benchmark=0.0)
    assert 0.0 <= psr <= 1.0


def test_dsr_single_trial() -> None:
    returns = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 200))
    dsr = deflated_sharpe(returns, np.array([0.0]))
    assert 0.0 <= dsr <= 1.0


def test_bootstrap_ruin() -> None:
    returns = pd.Series(np.random.default_rng(2).normal(-0.002, 0.02, 100))
    result = bootstrap_ruin_analysis(returns, starting_capital=5.0, n_samples=100, seed=1)
    assert 0.0 <= result.ruin_probability <= 1.0
    assert result.max_drawdown_p95 <= 0.0


def test_pbo_stub_raises() -> None:
    with pytest.raises(NotImplementedError):
        probability_backtest_overfitting([])


def test_max_drawdown() -> None:
    equity = pd.Series([100, 110, 90, 95])
    assert max_drawdown(equity) == pytest.approx(-0.1818, rel=0.01)
