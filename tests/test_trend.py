from __future__ import annotations

import numpy as np
import pandas as pd

from fdq.strategies.trend import Donchian, MACrossover


def _noisy_uptrend() -> pd.DataFrame:
    """Uptrend with real day-to-day noise so realized vol is meaningfully positive."""
    idx = pd.date_range("2019-01-01", periods=300, freq="B")
    rng = np.random.default_rng(7)
    drift = 100 * (1.0007 ** np.arange(300))
    close = pd.Series(drift * (1.0 + rng.normal(0.0, 0.01, 300)), index=idx)
    frame = pd.DataFrame({"close": close, "close_adj": close, "high": close + 0.5,
                          "low": close - 0.5, "open": close, "volume": 1e6})
    return pd.concat({"SPY": frame}, axis=1)


def _breakout_bars() -> pd.DataFrame:
    """Strong monotonic uptrend with high=close so the last bar is a fresh N-day high."""
    idx = pd.date_range("2019-01-01", periods=300, freq="B")
    close = pd.Series(100 * (1.003 ** np.arange(300)), index=idx)
    frame = pd.DataFrame({"close": close, "close_adj": close, "high": close,
                          "low": close, "open": close, "volume": 1e6})
    return pd.concat({"SPY": frame}, axis=1)


def test_ma_crossover_long_in_uptrend() -> None:
    bars = _noisy_uptrend()
    strat = MACrossover({"symbol": "SPY", "fast": 20, "slow": 50})
    asof = bars.index[-1].date()
    strat.should_rebalance(asof, bars)
    w = strat.target_weights(asof, bars)
    assert w.get("SPY", 0.0) == 1.0


def test_ma_crossover_flat_without_enough_history() -> None:
    bars = _noisy_uptrend()
    strat = MACrossover({"symbol": "SPY", "fast": 20, "slow": 50})
    asof = bars.index[10].date()
    strat.should_rebalance(asof, bars)
    w = strat.target_weights(asof, bars)
    assert w.empty or w.get("SPY", 0.0) == 0.0


def test_vol_filter_blocks_entry() -> None:
    bars = _noisy_uptrend()
    strat = MACrossover({"symbol": "SPY", "fast": 20, "slow": 50, "vol_max": 0.0001})
    asof = bars.index[-1].date()
    strat.should_rebalance(asof, bars)
    assert strat.target_weights(asof, bars).get("SPY", 0.0) == 0.0


def test_donchian_enters_on_breakout() -> None:
    bars = _breakout_bars()
    strat = Donchian({"symbol": "SPY", "window": 20})
    asof = bars.index[-1].date()
    strat.should_rebalance(asof, bars)
    assert strat.target_weights(asof, bars).get("SPY", 0.0) == 1.0
