"""Provenance enforcement tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fdq.data.provenance import ProvenanceError, validate_provenance

FIXTURE = Path(__file__).parent / "fixtures" / "SPY.parquet"


def test_fixture_has_real_provenance() -> None:
    doc = validate_provenance(FIXTURE)
    assert doc["synthetic"] is False
    assert doc["source"] in {"alpaca", "yfinance", "fred"}
    assert doc["data_kind"] == "historical"


def test_missing_provenance_rejected(tmp_path: Path) -> None:
    fake = tmp_path / "FAKE.parquet"
    fake.write_bytes(b"not parquet")
    with pytest.raises(ProvenanceError, match="Missing provenance"):
        validate_provenance(fake)


def test_synthetic_flag_rejected(tmp_path: Path) -> None:
    fake = tmp_path / "FAKE.parquet"
    fake.write_bytes(b"x")
    meta = tmp_path / "FAKE.parquet.meta.json"
    meta.write_text(json.dumps({"source": "yfinance", "synthetic": True}))
    with pytest.raises(ProvenanceError, match="synthetic"):
        validate_provenance(fake)
