"""Refresh CI test fixtures from real historical market data (yfinance)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from fdq.data.bars import _fetch_yfinance
from fdq.data.provenance import write_provenance

FIXTURE_DIR = Path(__file__).parent.parent.parent / "tests" / "fixtures"

# Frozen real-historical slices for CI smoke tests (not synthetic).
FIXTURE_SYMBOLS = ["SPY", "TLT"]
FIXTURE_START = date(2020, 1, 1)
FIXTURE_END = date(2020, 12, 31)


def refresh_test_fixtures() -> None:
    """Download real historical OHLCV slices into tests/fixtures/."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    fetched = _fetch_yfinance(FIXTURE_SYMBOLS, FIXTURE_START, FIXTURE_END)
    for sym in FIXTURE_SYMBOLS:
        if sym not in fetched or fetched[sym].empty:
            msg = f"yfinance returned no data for fixture symbol {sym}"
            raise RuntimeError(msg)
        path = FIXTURE_DIR / f"{sym}.parquet"
        fetched[sym].to_parquet(path)
        write_provenance(
            path,
            source="yfinance",
            data_kind="historical",
            start=FIXTURE_START,
            end=FIXTURE_END,
            symbol=sym,
        )
