"""Derived feature store — computed once, cached, versioned."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fdq.util.settings import Settings

DERIVED_SCHEMA_VERSION = "1.0.0"


def _features_path(symbol: str, data_dir: Path) -> Path:
    return data_dir / "processed" / f"{symbol}_features.parquet"


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def build_symbol_features(symbol: str, bars: pd.DataFrame) -> pd.DataFrame:
    close_col = bars["close_adj"] if "close_adj" in bars.columns else bars["close"]
    close = pd.Series(close_col, index=bars.index, dtype=float)
    log_ret = pd.Series(np.log(close / close.shift(1)), index=bars.index, dtype=float)
    features = pd.DataFrame(index=bars.index)
    features["log_return"] = log_ret
    features["realized_vol_21"] = log_ret.rolling(21).std() * np.sqrt(252)
    features["atr_14"] = _atr(bars["high"], bars["low"], close, 14)
    features["drawdown"] = close / close.cummax() - 1.0
    roll_mean = log_ret.rolling(63).mean()
    roll_std = log_ret.rolling(63).std()
    features["rolling_sharpe_63"] = (roll_mean / roll_std) * np.sqrt(252)
    features.attrs["derived_schema_version"] = DERIVED_SCHEMA_VERSION
    return features


def build_features(symbols: list[str], data_dir: Path | None = None) -> None:
    settings = Settings()
    base = data_dir or settings.data_dir
    raw_dir = base / "raw"
    proc_dir = base / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    for sym in symbols:
        raw_path = raw_dir / f"{sym}.parquet"
        if not raw_path.exists():
            continue
        bars = pd.read_parquet(raw_path)
        feats = build_symbol_features(sym, bars)
        feats.to_parquet(_features_path(sym, base))
