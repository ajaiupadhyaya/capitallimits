from __future__ import annotations

import numpy as np
import pandas as pd

from fdq.strategies.meanrev import Bollinger, RSIReversion, ZScoreReversion


def _dip_bars() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    base = np.full(120, 100.0)
    base[-1] = 80.0  # sharp oversold dip on the last bar
    close = pd.Series(base, index=idx)
    frame = pd.DataFrame({"close": close, "close_adj": close, "high": close + 1,
                          "low": close - 1, "open": close, "volume": 1e6})
    return pd.concat({"SPY": frame}, axis=1)


def test_zscore_enters_long_on_dip() -> None:
    bars = _dip_bars()
    strat = ZScoreReversion({"symbol": "SPY", "window": 20, "entry_z": 1.5})
    asof = bars.index[-1].date()
    strat.should_rebalance(asof, bars)
    assert strat.target_weights(asof, bars).get("SPY", 0.0) == 1.0


def test_rsi_enters_long_when_oversold() -> None:
    bars = _dip_bars()
    strat = RSIReversion({"symbol": "SPY", "period": 14, "low": 30.0, "high": 70.0})
    asof = bars.index[-1].date()
    strat.should_rebalance(asof, bars)
    assert strat.target_weights(asof, bars).get("SPY", 0.0) == 1.0


def test_bollinger_enters_below_lower_band() -> None:
    bars = _dip_bars()
    strat = Bollinger({"symbol": "SPY", "window": 20, "n_std": 2.0})
    asof = bars.index[-1].date()
    strat.should_rebalance(asof, bars)
    assert strat.target_weights(asof, bars).get("SPY", 0.0) == 1.0
