from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from fdq.experiment.runner import run_experiment

FIXTURES = Path(__file__).parent / "fixtures"


def _make_data_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    for name in ("SPY.parquet", "SPY.parquet.meta.json"):
        shutil.copy(FIXTURES / name, raw / name)
    return tmp_path / "data"


def test_walkforward_experiment_produces_dsr_and_pbo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _make_data_dir(tmp_path)
    monkeypatch.setenv("FDQ_DATA_DIR", str(data_dir))

    cfg = {
        "id": "EXP-TEST",
        "title": "walk-forward smoke",
        "friction_version": "1.0.0",
        "mode": "walkforward",
        "n_folds": 3,
        "strategies": [
            {"ma_crossover": {"symbol": "SPY", "grid": {"fast": [10, 20], "slow": [50]}}},
        ],
        "capital_tiers": [5, 50],
        "window": {"start": "2020-01-01", "end": "2020-12-31"},
        "stress_multipliers": [1],
        "seed": 42,
    }
    exp_dir = tmp_path / "EXP-TEST"
    exp_dir.mkdir()
    (exp_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    run_experiment(exp_dir / "config.yaml", tearsheet=False, report=True)

    summary = json.loads((exp_dir / "results" / "summary.json").read_text())
    assert summary["mode"] == "walkforward"
    tier5 = summary["strategies"][0]["tiers"]["5"]
    assert "dsr" in tier5 and "pbo" in tier5
    assert 0.0 <= tier5["pbo"] <= 1.0
    assert tier5["n_trials"] >= 3
    assert (exp_dir / "report.md").exists()
    assert "PBO" in (exp_dir / "report.md").read_text()
