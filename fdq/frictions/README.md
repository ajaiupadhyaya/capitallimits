# Friction Model v1.0.0

Retail broker emulator mirroring documented Alpaca behavior for micro-capital simulation.

## Assumptions

- **Minimum notional:** $1.00 per order; orders below this are rejected and logged.
- **Spread:** Full half-spread charged on entry and exit (no price improvement). Per-symbol bps table with VIX regime widening (1.5× when VIX > 25).
- **Fractional shares:** Dollar-notional positions; long-only.
- **Regulatory fees:** SEC fee + FINRA TAF on sells, with per-trade minimums (~$0.01 each). At $1 tickets these minimums dominate (~100bp).
- **Settlement:** T+1 cash account; sale proceeds unavailable until next business day.
- **Stress tiers:** All friction components scale linearly with `stress_multiplier` (1×, 2×, 5×).

## Versioning

Every backtest result must cite `friction_model_version`. Bump version when fee schedule or spread model changes.
