from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from fdq.strategies.signals import price_series, realized_vol, rolling_zscore, rsi, sma


def _bars() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    close = pd.Series(np.linspace(100, 160, 60), index=idx)
    frame = pd.DataFrame({"close": close, "close_adj": close, "high": close + 1,
                          "low": close - 1, "open": close, "volume": 1e6})
    return pd.concat({"SPY": frame}, axis=1)


def test_price_series_prefers_adjusted_and_truncates() -> None:
    bars = _bars()
    s = price_series(bars, "SPY", asof=date(2020, 1, 15))
    assert s.index.max().date() <= date(2020, 1, 15)
    assert s.iloc[0] == 100.0


def test_sma_matches_rolling_mean() -> None:
    bars = _bars()
    s = price_series(bars, "SPY")
    assert abs(sma(s, 5).iloc[-1] - s.iloc[-5:].mean()) < 1e-9


def test_rsi_bounds_and_zscore_and_vol() -> None:
    bars = _bars()
    s = price_series(bars, "SPY")
    r = rsi(s, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    assert abs(rolling_zscore(s, 10).iloc[-1]) < 5
    assert realized_vol(s, 21).dropna().iloc[-1] >= 0
