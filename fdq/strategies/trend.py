"""Tier-1 trend strategies — long/flat, single ETF. Pure signal logic."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from fdq.strategies.base import Strategy, StrategySpec
from fdq.strategies.signals import price_series, realized_vol, sma


class _LongFlat(Strategy):
    """Shared long/flat state machine keyed on a boolean desired state."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.symbol = str(self.params.get("symbol", "SPY"))
        vm = self.params.get("vol_max")
        self.vol_max: float | None = float(vm) if vm is not None else None  # type: ignore[arg-type]
        self._current = False
        self._pending = False
        self.spec = StrategySpec(self.slug(), self.name(), self.name(), [self.symbol])

    def slug(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def name(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def _desired(self, asof: date, bars: pd.DataFrame) -> bool:
        raise NotImplementedError

    def _vol_ok(self, asof: date, bars: pd.DataFrame) -> bool:
        if self.vol_max is None:
            return True
        close = price_series(bars, self.symbol, asof=asof)
        vol = realized_vol(close, 21).dropna()
        if vol.empty:
            return False
        return bool(vol.iloc[-1] <= self.vol_max)

    def should_rebalance(self, asof: date, bars: pd.DataFrame) -> bool:
        desired = self._desired(asof, bars)
        if desired and not self._vol_ok(asof, bars):
            desired = False
        if desired != self._current:
            self._pending = desired
            self._current = desired
            return True
        self._pending = self._current
        return False

    def target_weights(self, asof: date, bars: pd.DataFrame) -> pd.Series:
        if self._pending:
            return pd.Series({self.symbol: 1.0})
        return pd.Series(dtype=float)


class MACrossover(_LongFlat):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        p = params or {}
        self.fast = int(p.get("fast", 20))
        self.slow = int(p.get("slow", 50))
        super().__init__(params)

    def slug(self) -> str:
        return "ma_crossover"

    def name(self) -> str:
        return f"MA {self.fast}/{self.slow} {self.symbol}"

    def _desired(self, asof: date, bars: pd.DataFrame) -> bool:
        close = price_series(bars, self.symbol, asof=asof)
        if len(close) < self.slow + 1:
            return False
        fast_ma = sma(close, self.fast).iloc[-1]
        slow_ma = sma(close, self.slow).iloc[-1]
        return bool(fast_ma > slow_ma)


class Donchian(_LongFlat):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        p = params or {}
        self.window = int(p.get("window", 20))
        super().__init__(params)

    def slug(self) -> str:
        return "donchian"

    def name(self) -> str:
        return f"Donchian {self.window} {self.symbol}"

    def _desired(self, asof: date, bars: pd.DataFrame) -> bool:
        high = price_series(bars, self.symbol, field="high", asof=asof)
        low = price_series(bars, self.symbol, field="low", asof=asof)
        close = price_series(bars, self.symbol, asof=asof)
        if len(close) < self.window + 1:
            return self._current
        upper = high.iloc[-self.window - 1 : -1].max()
        lower = low.iloc[-self.window - 1 : -1].min()
        px = close.iloc[-1]
        if px >= upper:
            return True
        if px <= lower:
            return False
        return self._current
