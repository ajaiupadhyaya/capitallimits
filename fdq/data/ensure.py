"""Ensure real historical market data is cached before backtests or experiments."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from fdq.data.bars import BarRequest, _cache_path, fetch_symbol, get_bars
from fdq.data.macro import _macro_path, fetch_macro, load_macro
from fdq.data.provenance import ProvenanceError, validate_provenance
from fdq.util.settings import Settings


def ensure_symbol_cached(
    symbol: str,
    start: date,
    end: date,
    *,
    settings: Settings | None = None,
    auto_fetch: bool = True,
) -> None:
    settings = settings or Settings()
    path = _cache_path(symbol, settings.data_dir)
    if not path.exists():
        if not auto_fetch:
            msg = (
                f"No cached data for {symbol}. Run: "
                f"fdq data fetch --start {start} --end {end}"
            )
            raise ProvenanceError(msg)
        fetch_symbol(symbol, start, end, settings)
    validate_provenance(path)
    df = path.read_bytes()
    if len(df) < 100:
        msg = f"Cached data for {symbol} appears empty"
        raise ProvenanceError(msg)


def ensure_universe_cached(
    symbols: list[str],
    start: date,
    end: date,
    *,
    settings: Settings | None = None,
    auto_fetch: bool = True,
) -> None:
    for sym in symbols:
        ensure_symbol_cached(sym, start, end, settings=settings, auto_fetch=auto_fetch)


def ensure_macro_cached(
    start: date,
    end: date,
    *,
    settings: Settings | None = None,
    auto_fetch: bool = True,
) -> None:
    settings = settings or Settings()
    path = _macro_path(settings.data_dir)
    if not path.exists():
        if not auto_fetch:
            msg = "No cached macro data. Run: fdq data fetch"
            raise ProvenanceError(msg)
        fetch_macro(start, end, settings)
    validate_provenance(path)


def load_validated_bars(req: BarRequest, data_dir: Path | None = None) -> pd.DataFrame:
    settings = Settings()
    base = data_dir or settings.data_dir
    ensure_universe_cached(req.symbols, req.start, req.end, settings=settings, auto_fetch=True)
    bars = get_bars(req, data_dir=base)
    if bars.empty:
        msg = f"No bars for {req.symbols} in [{req.start}, {req.end}]"
        raise ProvenanceError(msg)
    return bars


def load_validated_macro(data_dir: Path | None = None) -> pd.DataFrame:
    settings = Settings()
    path = _macro_path(data_dir or settings.data_dir)
    if path.exists():
        validate_provenance(path)
    return load_macro(data_dir)
