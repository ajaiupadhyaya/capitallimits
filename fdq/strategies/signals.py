"""Pure signal helpers shared across strategy modules. No I/O."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def price_series(
    bars: pd.DataFrame, symbol: str, field: str = "close_adj", asof: date | None = None
) -> pd.Series:
    col = (symbol, field)
    if col not in bars.columns:
        col = (symbol, "close")
    s = pd.Series(bars[col], dtype=float).dropna()
    if asof is not None:
        s = s.loc[s.index <= pd.Timestamp(asof)]
    return s


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).mean()


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window).mean()
    std = s.rolling(window).std(ddof=1)
    return (s - mean) / std.replace(0.0, np.nan)


def rsi(s: pd.Series, period: int) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0.0).rolling(period).mean()
    loss = (-delta.clip(upper=0.0)).rolling(period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def realized_vol(close: pd.Series, window: int = 21) -> pd.Series:
    log_ret = pd.Series(np.log(close / close.shift(1)), index=close.index, dtype=float)
    vol = log_ret.rolling(window).std(ddof=1) * float(np.sqrt(252))
    return pd.Series(vol, index=close.index, dtype=float)
