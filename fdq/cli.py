"""Five Dollar Quant CLI."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.group()
def cli() -> None:
    """Five Dollar Quant — micro-capital simulation research platform."""


@cli.group()
def data() -> None:
    """Data ingestion and cache management."""


@data.command("fetch")
@click.option("--universe", default="config/universe.yaml", type=click.Path(exists=True))
@click.option("--start", default="2016-06-01")
@click.option("--end", default="2026-06-01")
def data_fetch(universe: str, start: str, end: str) -> None:
    """Fetch OHLCV bars and macro series into parquet cache."""
    from datetime import date

    from fdq.data.bars import fetch_universe
    from fdq.data.macro import fetch_macro
    from fdq.util.settings import Settings

    symbols = _load_universe(universe)
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    fetch_universe(symbols, start_d, end_d)
    settings = Settings()
    if settings.fred_api_key:
        fetch_macro(start_d, end_d)
        console.print(f"[green]Fetched {len(symbols)} symbols + FRED macro[/green]")
    else:
        console.print(
            f"[green]Fetched {len(symbols)} symbols[/green] "
            "[yellow](skipped macro — set FRED_API_KEY in .env)[/yellow]"
        )


@data.command("build-features")
@click.option("--universe", default="config/universe.yaml", type=click.Path(exists=True))
def data_build_features(universe: str) -> None:
    """Compute derived feature store from cached bars."""
    from fdq.data.derived import build_features

    symbols = _load_universe(universe)
    build_features(symbols)
    console.print(f"[green]Built features for {len(symbols)} symbols[/green]")


@data.command("refresh-fixtures")
def data_refresh_fixtures() -> None:
    """Re-download real historical slices for CI test fixtures (yfinance)."""
    from fdq.data.fixtures import refresh_test_fixtures

    refresh_test_fixtures()
    console.print("[green]Refreshed tests/fixtures from real historical data[/green]")


@data.command("doctor")
@click.option("--universe", default="config/universe.yaml", type=click.Path(exists=True))
def data_doctor(universe: str) -> None:
    """Report cache coverage and data quality."""
    from fdq.data.doctor import run_doctor

    symbols = _load_universe(universe)
    run_doctor(symbols)


@cli.command("backtest")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def backtest_run(config_path: str) -> None:
    """Run a single backtest from experiment config."""
    from fdq.experiment.runner import run_experiment

    run_experiment(Path(config_path), tearsheet=False, report=False)


@cli.group()
def experiment() -> None:
    """Experiment pipeline commands."""


@experiment.command("run")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--no-tearsheet", is_flag=True, default=False)
@click.option("--no-report", is_flag=True, default=False)
def experiment_run(config_path: str, no_tearsheet: bool, no_report: bool) -> None:
    """Run full experiment pipeline: backtest + tearsheet + report."""
    from fdq.experiment.runner import run_experiment

    run_experiment(
        Path(config_path),
        tearsheet=not no_tearsheet,
        report=not no_report,
    )


@cli.command("smoke")
def smoke() -> None:
    """CI smoke test: minimal backtest on real historical fixture slices."""
    from fdq.smoke import run_smoke

    run_smoke()
    console.print("[green]Smoke test passed[/green]")


def _load_universe(path: str) -> list[str]:
    import yaml

    with open(path) as f:
        doc = yaml.safe_load(f)
    return list(doc["symbols"])
