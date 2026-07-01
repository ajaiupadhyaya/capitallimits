"""Tier 0 benchmark strategies."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from fdq.strategies.base import Strategy, StrategySpec


class BuyAndHold(Strategy):
    spec = StrategySpec(
        slug="buy_and_hold",
        name="Buy and Hold",
        description="100% single ETF from day 1, hold",
        universe=[],
    )

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.symbol: str = str(self.params.get("symbol", "SPY"))
        self.spec = StrategySpec(
            slug="buy_and_hold",
            name=f"Buy and Hold {self.symbol}",
            description=f"100% {self.symbol} from day 1",
            universe=[self.symbol],
        )
        self._initialized = False

    def target_weights(self, asof: date, bars: pd.DataFrame) -> pd.Series:
        return pd.Series({self.symbol: 1.0})

    def should_rebalance(self, asof: date, bars: pd.DataFrame) -> bool:
        if self._initialized:
            return False
        self._initialized = True
        return True


class Balanced6040(Strategy):
    spec = StrategySpec(
        slug="balanced_6040",
        name="60/40 SPY/TLT",
        description="60% SPY / 40% TLT monthly rebalance",
        universe=["SPY", "TLT"],
    )

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        syms_raw = self.params.get("symbols", ["SPY", "TLT"])
        syms = list(syms_raw) if isinstance(syms_raw, (list, tuple)) else ["SPY", "TLT"]
        self.spy = str(syms[0])
        self.tlt = str(syms[1])
        self.weights = {self.spy: 0.6, self.tlt: 0.4}
        self._last_rebalance_month: tuple[int, int] | None = None

    def target_weights(self, asof: date, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(self.weights)

    def should_rebalance(self, asof: date, bars: pd.DataFrame) -> bool:
        month_key = (asof.year, asof.month)
        if self._last_rebalance_month is None:
            self._last_rebalance_month = month_key
            return True
        if month_key != self._last_rebalance_month:
            self._last_rebalance_month = month_key
            return True
        return False


def build_strategy(name: str, params: dict[str, Any]) -> Strategy:
    from fdq.strategies.meanrev import Bollinger, RSIReversion, ZScoreReversion
    from fdq.strategies.trend import Donchian, MACrossover

    registry: dict[str, type[Strategy]] = {
        "buy_and_hold": BuyAndHold,
        "balanced_6040": Balanced6040,
        "ma_crossover": MACrossover,
        "donchian": Donchian,
        "zscore_reversion": ZScoreReversion,
        "rsi_reversion": RSIReversion,
        "bollinger": Bollinger,
    }
    if name not in registry:
        msg = f"Unknown strategy: {name}"
        raise ValueError(msg)
    return registry[name](params)
