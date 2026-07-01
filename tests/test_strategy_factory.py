from __future__ import annotations

import pytest

from fdq.strategies.benchmarks import build_strategy


@pytest.mark.parametrize(
    "name,params",
    [
        ("ma_crossover", {"symbol": "SPY", "fast": 20, "slow": 50}),
        ("donchian", {"symbol": "SPY", "window": 20}),
        ("zscore_reversion", {"symbol": "SPY", "window": 20, "entry_z": 1.5}),
        ("rsi_reversion", {"symbol": "SPY", "period": 14, "low": 30, "high": 70}),
        ("bollinger", {"symbol": "SPY", "window": 20, "n_std": 2.0}),
    ],
)
def test_factory_builds_tier1(name: str, params: dict) -> None:
    strat = build_strategy(name, params)
    assert strat.spec.slug == name


def test_factory_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        build_strategy("nope", {})
