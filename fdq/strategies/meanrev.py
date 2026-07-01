"""Tier-1 mean-reversion strategies — long/flat, single ETF. Pure signal logic."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from fdq.strategies.signals import price_series, rolling_zscore, rsi, sma
from fdq.strategies.trend import _LongFlat


class ZScoreReversion(_LongFlat):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        p = params or {}
        self.window = int(p.get("window", 20))
        self.entry_z = float(p.get("entry_z", 1.5))
        self.exit_z = float(p.get("exit_z", 0.0))
        super().__init__(params)

    def slug(self) -> str:
        return "zscore_reversion"

    def name(self) -> str:
        return f"Z-Reversion {self.window} {self.symbol}"

    def _desired(self, asof: date, bars: pd.DataFrame) -> bool:
        close = price_series(bars, self.symbol, asof=asof)
        if len(close) < self.window + 1:
            return self._current
        z = rolling_zscore(close, self.window).iloc[-1]
        if pd.isna(z):
            return self._current
        if z <= -self.entry_z:
            return True
        if z >= self.exit_z:
            return False
        return self._current


class RSIReversion(_LongFlat):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        p = params or {}
        self.period = int(p.get("period", 14))
        self.low = float(p.get("low", 30.0))
        self.high = float(p.get("high", 70.0))
        super().__init__(params)

    def slug(self) -> str:
        return "rsi_reversion"

    def name(self) -> str:
        return f"RSI {self.period} {self.symbol}"

    def _desired(self, asof: date, bars: pd.DataFrame) -> bool:
        close = price_series(bars, self.symbol, asof=asof)
        if len(close) < self.period + 1:
            return self._current
        val = rsi(close, self.period).iloc[-1]
        if pd.isna(val):
            return self._current
        if val <= self.low:
            return True
        if val >= self.high:
            return False
        return self._current


class Bollinger(_LongFlat):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        p = params or {}
        self.window = int(p.get("window", 20))
        self.n_std = float(p.get("n_std", 2.0))
        super().__init__(params)

    def slug(self) -> str:
        return "bollinger"

    def name(self) -> str:
        return f"Bollinger {self.window}/{self.n_std} {self.symbol}"

    def _desired(self, asof: date, bars: pd.DataFrame) -> bool:
        close = price_series(bars, self.symbol, asof=asof)
        if len(close) < self.window + 1:
            return self._current
        mid = sma(close, self.window).iloc[-1]
        std = close.rolling(self.window).std(ddof=1).iloc[-1]
        px = close.iloc[-1]
        if pd.isna(mid) or pd.isna(std):
            return self._current
        if px <= mid - self.n_std * std:
            return True
        if px >= mid:
            return False
        return self._current
