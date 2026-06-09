# capitallimits / Five Dollar Quant

Simulation-first study of institutional quantitative methods under extreme capital constraints ($5 starting capital, realistic retail frictions).

## Data policy

**No synthetic market data.** All backtests and experiments use:

- **Historical:** Alpaca (primary) or yfinance (fallback) for OHLCV; FRED for macro
- **Live (Phase 4+):** Alpaca paper/live API only

Every cached parquet file has a `.meta.json` provenance sidecar. Data without valid provenance is rejected.

## Setup

```bash
uv sync --all-extras
cp .env.example .env   # ALPACA_API_KEY, ALPACA_SECRET_KEY, FRED_API_KEY
```

## Data

```bash
fdq data fetch --universe config/universe.yaml --start 2016-06-01 --end 2026-06-01
fdq data build-features
fdq data doctor
```

`fdq data fetch` pulls **real** historical bars. Alpaca is used when keys are set; otherwise yfinance.

## Run EXP-001

```bash
fdq data fetch   # required before first run
fdq experiment run experiments/EXP-001/config.yaml
```

Outputs: `experiments/EXP-001/results/`, `report.md`, `tearsheet.html`

## Development

```bash
fdq data refresh-fixtures   # re-download real CI fixture slices
uv run pytest
uv run fdq smoke
```

## Documentation

- [updatedplan.md](updatedplan.md) — v2.0 research plan (simulation-first)
- [handoff.md](handoff.md) — v1.0 handoff (superseded)
