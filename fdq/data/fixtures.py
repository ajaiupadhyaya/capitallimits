"""Refresh CI test fixtures from real historical market data.

Prefers slicing the local real cache (``data/raw``) when present — deterministic
and offline — and falls back to a fresh yfinance download otherwise. Either way the
fixtures are real market data with valid provenance; never synthetic.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from fdq.data.bars import _cache_path, _fetch_yfinance
from fdq.data.provenance import read_provenance, write_provenance
from fdq.util.settings import Settings

FIXTURE_DIR = Path(__file__).parent.parent.parent / "tests" / "fixtures"

# Frozen real-historical slices for CI tests (not synthetic).
# ~750 trading days so walk-forward folds are statistically meaningful.
FIXTURE_SYMBOLS = ["SPY", "TLT"]
FIXTURE_START = date(2018, 1, 1)
FIXTURE_END = date(2020, 12, 31)


def _slice_from_cache(symbol: str) -> tuple[pd.DataFrame, str] | None:
    """Return (sliced bars, source) from the local real cache, if available."""
    settings = Settings()
    path = _cache_path(symbol, settings.data_dir)
    if not path.exists():
        return None
    doc = read_provenance(path)
    source = str(doc.get("source", "alpaca"))
    df = pd.read_parquet(path)
    mask = (df.index >= pd.Timestamp(FIXTURE_START)) & (df.index <= pd.Timestamp(FIXTURE_END))
    sliced = df.loc[mask]
    if sliced.empty:
        return None
    return sliced, source


def refresh_test_fixtures() -> None:
    """Write real historical OHLCV slices into tests/fixtures/ with provenance."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for sym in FIXTURE_SYMBOLS:
        cached = _slice_from_cache(sym)
        if cached is not None:
            df, source = cached
        else:
            fetched = _fetch_yfinance([sym], FIXTURE_START, FIXTURE_END)
            if sym not in fetched or fetched[sym].empty:
                msg = f"No real data available for fixture symbol {sym}"
                raise RuntimeError(msg)
            df, source = fetched[sym], "yfinance"
        path = FIXTURE_DIR / f"{sym}.parquet"
        df.to_parquet(path)
        write_provenance(
            path,
            source=source,  # type: ignore[arg-type]
            data_kind="historical",
            start=FIXTURE_START,
            end=FIXTURE_END,
            symbol=sym,
        )
