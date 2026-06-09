"""Backtest engine tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from fdq.backtest.engine import BacktestConfig, run_backtest, run_capital_sweep
from fdq.data.provenance import validate_provenance
from fdq.frictions.config import FrictionConfig
from fdq.strategies.benchmarks import Balanced6040, BuyAndHold

FIXTURE = Path(__file__).parent / "fixtures" / "SPY.parquet"


@pytest.fixture(autouse=True)
def _validate_fixture_provenance() -> None:
    validate_provenance(FIXTURE)
    tlt = Path(__file__).parent / "fixtures" / "TLT.parquet"
    if tlt.exists():
        validate_provenance(tlt)


@pytest.fixture
def bars() -> pd.DataFrame:
    spy = pd.read_parquet(FIXTURE)
    tlt_path = Path(__file__).parent / "fixtures" / "TLT.parquet"
    if tlt_path.exists():
        tlt = pd.read_parquet(tlt_path)
        return pd.concat({"SPY": spy, "TLT": tlt}, axis=1)
    return pd.concat({"SPY": spy}, axis=1)


def test_buy_and_hold_runs(bars: pd.DataFrame) -> None:
    strategy = BuyAndHold({"symbol": "SPY"})
    config = BacktestConfig(starting_capital=5.0, friction=FrictionConfig())
    result = run_backtest(strategy, bars, config, start=date(2020, 1, 1), end=date(2020, 12, 31))
    assert len(result.equity_curve) > 50
    assert result.ending_equity > 4.0
    assert len(result.trades) >= 1
    assert result.cost_ledger.spread_cents > 0


def test_balanced_6040_rebalances(bars: pd.DataFrame) -> None:
    if ("TLT", "close") not in bars.columns and "TLT" not in str(bars.columns):
        pytest.skip("TLT fixture missing")
    strategy = Balanced6040({"symbols": ["SPY", "TLT"]})
    config = BacktestConfig(starting_capital=50.0, friction=FrictionConfig())
    result = run_backtest(strategy, bars, config, start=date(2020, 1, 1), end=date(2020, 6, 30))
    assert len(result.trades) >= 2


def test_capital_sweep_fresh_strategy_per_tier(bars: pd.DataFrame) -> None:
    tiers = [5.0, 50.0]
    results = run_capital_sweep(
        lambda: BuyAndHold({"symbol": "SPY"}),
        bars,
        tiers,
        FrictionConfig(),
        start=date(2020, 1, 1),
        end=date(2020, 6, 30),
    )
    assert len(results[5.0].trades) >= 1
    assert len(results[50.0].trades) >= 1


def test_metadata_includes_friction_version(bars: pd.DataFrame) -> None:
    strategy = BuyAndHold({"symbol": "SPY"})
    result = run_backtest(strategy, bars, BacktestConfig(starting_capital=5.0), start=date(2020, 3, 1), end=date(2020, 4, 30))
    assert result.metadata["friction_model_version"] == "1.0.0"
