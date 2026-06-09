"""FRED macro series cache."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from fdq.data.provenance import write_provenance
from fdq.util.settings import Settings

FRED_SERIES = {
    "VIXCLS": "vix",
    "DGS10": "dgs10",
    "DGS2": "dgs2",
}


def _macro_path(data_dir: Path) -> Path:
    return data_dir / "raw" / "macro.parquet"


def fetch_macro(start: date, end: date, settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or Settings()
    if not settings.fred_api_key:
        msg = "FRED_API_KEY required to fetch real macro data. Set it in .env."
        raise ValueError(msg)
    from fredapi import Fred

    fred = Fred(api_key=settings.fred_api_key)
    frames: dict[str, pd.Series] = {}
    for series_id, col in FRED_SERIES.items():
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
        frames[col] = s.rename(col)

    df = pd.DataFrame(frames)
    df.index = pd.DatetimeIndex(df.index, name="timestamp")
    df = df.sort_index().ffill()
    path = _macro_path(settings.data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    write_provenance(
        path,
        source="fred",
        data_kind="historical",
        start=start,
        end=end,
        series=list(FRED_SERIES.values()),
    )
    return df


def load_macro(data_dir: Path | None = None) -> pd.DataFrame:
    settings = Settings()
    path = _macro_path(data_dir or settings.data_dir)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
