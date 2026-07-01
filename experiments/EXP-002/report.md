# EXP-002 — How much trend-following alpha survives friction at $5?

## Methodology

- Mode: walk-forward (4 expanding folds), grid-search on train only
- Window (in-sample): 2016-06-01 → 2022-05-31
- Friction model version: 1.0.0
- Every reported result is net-of-friction; DSR deflates by trial count; PBO via CSCV.

## Results

### ma_crossover ({'symbol': 'SPY'})

| Tier | CAGR | Sharpe | Max DD | DSR | PBO | Trials |
|------|------|--------|--------|-----|-----|--------|
| $5 | 5.78% | 0.55 | -16.62% | 0.637 | 0.725 | 36 |
| $50 | 7.31% | 0.67 | -16.06% | 0.727 | 0.725 | 36 |
| $500 | 7.38% | 0.67 | -16.00% | 0.731 | 0.725 | 36 |
| $5000 | 7.39% | 0.67 | -16.00% | 0.731 | 0.725 | 36 |
| $50000 | 7.39% | 0.67 | -16.00% | 0.731 | 0.725 | 36 |

### donchian ({'symbol': 'SPY'})

| Tier | CAGR | Sharpe | Max DD | DSR | PBO | Trials |
|------|------|--------|--------|-----|-----|--------|
| $5 | 6.96% | 0.71 | -13.41% | 0.889 | 0.253 | 8 |
| $50 | 7.72% | 0.78 | -12.83% | 0.915 | 0.253 | 8 |
| $500 | 7.80% | 0.79 | -12.77% | 0.917 | 0.253 | 8 |
| $5000 | 7.80% | 0.79 | -12.77% | 0.917 | 0.253 | 8 |
| $50000 | 7.80% | 0.79 | -12.77% | 0.917 | 0.253 | 8 |

### ma_crossover ({'symbol': 'QQQ'})

| Tier | CAGR | Sharpe | Max DD | DSR | PBO | Trials |
|------|------|--------|--------|-----|-----|--------|
| $5 | 10.57% | 0.66 | -25.94% | 0.705 | 0.777 | 36 |
| $50 | 11.21% | 0.70 | -25.94% | 0.728 | 0.777 | 36 |
| $500 | 11.27% | 0.70 | -25.94% | 0.731 | 0.777 | 36 |
| $5000 | 11.28% | 0.70 | -25.94% | 0.731 | 0.777 | 36 |
| $50000 | 11.28% | 0.70 | -25.94% | 0.731 | 0.777 | 36 |

### donchian ({'symbol': 'QQQ'})

| Tier | CAGR | Sharpe | Max DD | DSR | PBO | Trials |
|------|------|--------|--------|-----|-----|--------|
| $5 | 10.77% | 0.76 | -16.40% | 0.911 | 0.240 | 8 |
| $50 | 11.17% | 0.79 | -16.07% | 0.920 | 0.240 | 8 |
| $500 | 11.21% | 0.79 | -16.03% | 0.920 | 0.240 | 8 |
| $5000 | 11.21% | 0.79 | -16.03% | 0.920 | 0.240 | 8 |
| $50000 | 11.22% | 0.79 | -16.03% | 0.920 | 0.240 | 8 |

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
