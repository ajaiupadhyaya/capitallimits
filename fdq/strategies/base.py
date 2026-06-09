"""Strategy ABC — pure signal functions, no I/O."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    slug: str
    name: str
    description: str
    universe: list[str]


class Strategy(ABC):
    spec: StrategySpec

    def __init__(self, params: dict[str, object] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def target_weights(self, asof: date, bars: pd.DataFrame) -> pd.Series:
        """Return target portfolio weights summing to <= 1.0."""

    def target_dollars(
        self, asof: date, equity: float, bars: pd.DataFrame, deployable_pct: float = 0.9
    ) -> dict[str, float]:
        weights = self.target_weights(asof, bars)
        deployable = equity * deployable_pct
        return {sym: float(weights.get(sym, 0.0)) * deployable for sym in weights.index if weights[sym] > 0}

    def should_rebalance(self, asof: date, bars: pd.DataFrame) -> bool:
        return True
