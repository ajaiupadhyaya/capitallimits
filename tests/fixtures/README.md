# Test fixtures

Frozen **real historical** OHLCV slices for CI smoke tests. Not synthetic.

- Source: yfinance (same vendor used as production fallback)
- Refresh: `fdq data refresh-fixtures`

Each `.parquet` file has a sibling `.parquet.meta.json` provenance record.
