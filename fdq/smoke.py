"""CI smoke test using frozen real historical fixture slices."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from fdq.backtest.engine import BacktestConfig, run_backtest
from fdq.data.provenance import validate_provenance
from fdq.frictions.config import FrictionConfig
from fdq.strategies.benchmarks import BuyAndHold

FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


def _load_fixture_bars() -> pd.DataFrame:
    path = FIXTURE_DIR / "SPY.parquet"
    validate_provenance(path)
    spy = pd.read_parquet(path)
    return pd.concat({"SPY": spy}, axis=1)


def run_smoke() -> None:
    bars = _load_fixture_bars()
    strategy = BuyAndHold({"symbol": "SPY"})
    friction = FrictionConfig()
    config = BacktestConfig(starting_capital=5.0, friction=friction, seed=42)
    result = run_backtest(
        strategy,
        bars,
        config,
        start=date(2020, 1, 1),
        end=date(2020, 6, 30),
    )
    assert len(result.equity_curve) > 0, "equity curve empty"
    assert result.ending_equity > 0, "ruined immediately"
    assert result.metadata["friction_model_version"] == "1.0.0"
    assert len(result.trades) >= 1, "expected at least one trade"
