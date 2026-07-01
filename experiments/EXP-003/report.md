# EXP-003 — Does mean-reversion survive friction at $5?

## Methodology

- Mode: walk-forward (4 expanding folds), grid-search on train only
- Window (in-sample): 2016-06-01 → 2022-05-31
- Friction model version: 1.0.0
- Every reported result is net-of-friction; DSR deflates by trial count; PBO via CSCV.

## Results

### zscore_reversion ({'symbol': 'SPY'})

| Tier | CAGR | Sharpe | Max DD | DSR | PBO | Trials |
|------|------|--------|--------|-----|-----|--------|
| $5 | -2.06% | -0.08 | -27.29% | 0.231 | 0.552 | 36 |
| $50 | 3.31% | 0.31 | -28.14% | 0.546 | 0.552 | 36 |
| $500 | 3.65% | 0.33 | -28.14% | 0.567 | 0.552 | 36 |
| $5000 | 3.67% | 0.33 | -28.14% | 0.568 | 0.552 | 36 |
| $50000 | 3.67% | 0.33 | -28.14% | 0.568 | 0.552 | 36 |

### rsi_reversion ({'symbol': 'SPY'})

| Tier | CAGR | Sharpe | Max DD | DSR | PBO | Trials |
|------|------|--------|--------|-----|-----|--------|
| $5 | 2.07% | 0.22 | -26.57% | 0.575 | 0.500 | 12 |
| $50 | 2.84% | 0.27 | -26.32% | 0.620 | 0.500 | 12 |
| $500 | 2.92% | 0.27 | -26.30% | 0.624 | 0.500 | 12 |
| $5000 | 2.92% | 0.27 | -26.30% | 0.624 | 0.500 | 12 |
| $50000 | 2.92% | 0.27 | -26.30% | 0.624 | 0.500 | 12 |

### bollinger ({'symbol': 'SPY'})

| Tier | CAGR | Sharpe | Max DD | DSR | PBO | Trials |
|------|------|--------|--------|-----|-----|--------|
| $5 | 1.19% | 0.21 | -9.01% | 0.548 | 0.585 | 12 |
| $50 | 1.32% | 0.17 | -26.32% | 0.507 | 0.585 | 12 |
| $500 | 1.43% | 0.18 | -26.30% | 0.515 | 0.585 | 12 |
| $5000 | 1.44% | 0.18 | -26.29% | 0.515 | 0.585 | 12 |
| $50000 | 1.44% | 0.18 | -26.29% | 0.515 | 0.585 | 12 |

### zscore_reversion ({'symbol': 'IWM'})

| Tier | CAGR | Sharpe | Max DD | DSR | PBO | Trials |
|------|------|--------|--------|-----|-----|--------|
| $5 | -5.12% | -0.31 | -36.39% | 0.138 | 0.914 | 36 |
| $50 | -2.81% | -0.10 | -37.89% | 0.272 | 0.914 | 36 |
| $500 | -2.47% | -0.08 | -37.76% | 0.288 | 0.914 | 36 |
| $5000 | -2.45% | -0.08 | -37.76% | 0.289 | 0.914 | 36 |
| $50000 | -2.45% | -0.08 | -37.76% | 0.289 | 0.914 | 36 |

## Statistical power (pre-registered)

Annualized minimum detectable Sharpe (one-sided, alpha=0.05, power=0.80) — the smallest true Sharpe a forward run of this length could distinguish from zero:

| Horizon | Min. detectable Sharpe |
|---------|------------------------|
| 6 months | 3.52 |
| 12 months | 2.49 |
| 24 months | 1.76 |

Reported OOS Sharpes below these thresholds are not statistically resolvable over the corresponding horizon, independent of friction.

## Limitations

- Locked test set (2024-2026) not used here.
