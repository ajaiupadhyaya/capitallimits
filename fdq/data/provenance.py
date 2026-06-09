"""Data provenance — all market data must come from real historical or live sources."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

DataSource = Literal["alpaca", "yfinance", "fred"]
DataKind = Literal["historical", "live"]

ALLOWED_SOURCES: frozenset[str] = frozenset({"alpaca", "yfinance", "fred"})


class ProvenanceError(ValueError):
    """Raised when cached data lacks valid real-market provenance."""


def meta_path(parquet_path: Path) -> Path:
    return parquet_path.with_name(f"{parquet_path.name}.meta.json")


def write_provenance(
    parquet_path: Path,
    *,
    source: DataSource,
    data_kind: DataKind,
    start: date | None = None,
    end: date | None = None,
    symbol: str | None = None,
    series: list[str] | None = None,
) -> None:
    if source not in ALLOWED_SOURCES:
        msg = f"Refusing to record unknown data source: {source}"
        raise ProvenanceError(msg)

    doc: dict[str, object] = {
        "source": source,
        "data_kind": data_kind,
        "fetched_at": datetime.now(UTC).isoformat(),
        "synthetic": False,
    }
    if symbol is not None:
        doc["symbol"] = symbol
    if series is not None:
        doc["series"] = series
    if start is not None:
        doc["start"] = start.isoformat()
    if end is not None:
        doc["end"] = end.isoformat()

    path = meta_path(parquet_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(doc, f, indent=2)


def read_provenance(parquet_path: Path) -> dict[str, object]:
    path = meta_path(parquet_path)
    if not path.exists():
        msg = (
            f"Missing provenance for {parquet_path.name}. "
            "Re-fetch with `fdq data fetch` (real historical sources only)."
        )
        raise ProvenanceError(msg)
    with path.open() as f:
        doc: dict[str, object] = json.load(f)
    return doc


def validate_provenance(parquet_path: Path) -> dict[str, object]:
    doc = read_provenance(parquet_path)
    if doc.get("synthetic") is True:
        msg = f"Refusing synthetic data: {parquet_path}"
        raise ProvenanceError(msg)
    source = doc.get("source")
    if source not in ALLOWED_SOURCES:
        msg = f"Invalid or missing source in provenance for {parquet_path}: {source!r}"
        raise ProvenanceError(msg)
    return doc
