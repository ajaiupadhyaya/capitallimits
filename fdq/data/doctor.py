"""Data cache health report."""

from __future__ import annotations

import pandas as pd
from rich.console import Console
from rich.table import Table

from fdq.data.provenance import read_provenance
from fdq.util.settings import Settings

console = Console()


def run_doctor(symbols: list[str]) -> None:
    settings = Settings()
    table = Table(title="Data Cache Doctor")
    table.add_column("Symbol")
    table.add_column("Rows")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Source")
    table.add_column("Status")

    for sym in symbols:
        path = settings.raw_dir / f"{sym}.parquet"
        if not path.exists():
            table.add_row(sym, "—", "—", "—", "—", "[red]MISSING[/red]")
            continue
        df = pd.read_parquet(path)
        start = str(df.index.min().date()) if len(df) else "—"
        end = str(df.index.max().date()) if len(df) else "—"
        try:
            prov = read_provenance(path)
            source = str(prov.get("source", "?"))
        except Exception:
            source = "[red]no provenance[/red]"
        table.add_row(sym, str(len(df)), start, end, source, "[green]OK[/green]")

    macro_path = settings.raw_dir / "macro.parquet"
    if macro_path.exists():
        macro = pd.read_parquet(macro_path)
        try:
            prov = read_provenance(macro_path)
            msource = str(prov.get("source", "?"))
        except Exception:
            msource = "[red]no provenance[/red]"
        table.add_row(
            "MACRO",
            str(len(macro)),
            str(macro.index.min().date()),
            str(macro.index.max().date()),
            msource,
            "[green]OK[/green]",
        )
    else:
        table.add_row("MACRO", "—", "—", "—", "—", "[yellow]MISSING[/yellow]")

    disc_path = settings.quality_dir / "discrepancies.jsonl"
    if disc_path.exists():
        n = sum(1 for _ in disc_path.open())
        console.print(f"Price discrepancies logged: {n}")
    else:
        console.print("No price discrepancies logged.")

    console.print(table)
