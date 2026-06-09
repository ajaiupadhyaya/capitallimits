# EXP-001 — What does buy-and-hold cost at $5?

## Hypothesis

Even passive buy-and-hold on liquid ETFs incurs meaningful friction drag at $5 starting capital due to spread costs on entry and regulatory fee floors on exit. Monthly 60/40 rebalancing may fail the $1 minimum order constraint or bleed to fees.

## Data & Window

- Universe: Tier 0 ETFs
- Window: 2016-06-01 → 2022-05-31 (in-sample)
- Friction model version: 1.0.0

## Methodology

- Strategies: buy-and-hold (SPY, QQQ, IWM), 60/40 SPY/TLT monthly rebalance
- Execution: signal at close T, fill at open T+1
- Capital tiers: $5, $50, $500, $5000, $50000
- Trial count: 1 (no hyperparameter search)
- Stress multipliers: [1, 2, 5]

## Results

### buy_and_hold ({'symbol': 'SPY'})

| Tier | CAGR | Sharpe | Max DD | Total Cost (¢) | Trades | Rejections |
|------|------|--------|--------|----------------|--------|------------|
| $5 | 11.02% | 0.70 | -31.98% | 0.0 | 1 | 0 |
| $50 | 11.02% | 0.70 | -31.98% | 0.5 | 1 | 0 |
| $500 | 11.02% | 0.70 | -31.98% | 4.5 | 1 | 0 |
| $5000 | 11.02% | 0.70 | -31.98% | 45.0 | 1 | 0 |
| $50000 | 11.02% | 0.70 | -31.98% | 450.0 | 1 | 0 |

### buy_and_hold ({'symbol': 'QQQ'})

| Tier | CAGR | Sharpe | Max DD | Total Cost (¢) | Trades | Rejections |
|------|------|--------|--------|----------------|--------|------------|
| $5 | 17.42% | 0.86 | -28.05% | 0.1 | 1 | 0 |
| $50 | 17.42% | 0.86 | -28.05% | 0.7 | 1 | 0 |
| $500 | 17.42% | 0.86 | -28.05% | 6.7 | 1 | 0 |
| $5000 | 17.42% | 0.86 | -28.05% | 67.5 | 1 | 0 |
| $50000 | 17.42% | 0.86 | -28.05% | 675.0 | 1 | 0 |

### buy_and_hold ({'symbol': 'IWM'})

| Tier | CAGR | Sharpe | Max DD | Total Cost (¢) | Trades | Rejections |
|------|------|--------|--------|----------------|--------|------------|
| $5 | 7.51% | 0.44 | -39.47% | 0.1 | 1 | 0 |
| $50 | 7.51% | 0.44 | -39.47% | 0.9 | 1 | 0 |
| $500 | 7.51% | 0.44 | -39.47% | 9.0 | 1 | 0 |
| $5000 | 7.51% | 0.44 | -39.47% | 90.0 | 1 | 0 |
| $50000 | 7.51% | 0.44 | -39.47% | 900.0 | 1 | 0 |

### balanced_6040 ({'symbols': ['SPY', 'TLT']})

| Tier | CAGR | Sharpe | Max DD | Total Cost (¢) | Trades | Rejections |
|------|------|--------|--------|----------------|--------|------------|
| $5 | 6.20% | 0.64 | -19.50% | 2.1 | 4 | 141 |
| $50 | 6.20% | 0.69 | -17.57% | 51.7 | 48 | 121 |
| $500 | 6.29% | 0.70 | -17.51% | 150.7 | 129 | 80 |
| $5000 | 6.32% | 0.70 | -17.50% | 345.9 | 143 | 70 |
| $50000 | 6.32% | 0.70 | -17.50% | 2307.8 | 144 | 70 |

## Statistical Assessment

- DSR with trial_count=1 reduces to PSR vs zero benchmark
- PBO/CSCV: N/A for Tier 0 (no parameter search)
- Bootstrap ruin analysis at $5 tier reported in summary.json

## Capital-Viability Finding

See tearsheet.html for capital-viability frontier (Sharpe vs starting capital, log scale).
Break-even capital is the tier where net Sharpe crosses zero after friction.

## Limitations & Threats to Validity

- Spread model uses static bps estimates, not historical NBBO
- No market impact model (irrelevant at micro scale)
- Locked test set (2024-2026) not used in this experiment

## Conclusion & Next Steps

EXP-001 establishes the friction baseline for passive strategies. Phase 1 will add Tier 1 rule-based strategies with walk-forward validation.
