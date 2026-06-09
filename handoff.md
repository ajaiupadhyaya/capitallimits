# PROJECT HANDOFF

# FIVE DOLLAR QUANT

## Institutional-Grade Quantitative Research System Operating Under Extreme Capital Constraints

Author: Ajai Upadhyaya

Version: 1.0

---

# EXECUTIVE SUMMARY

This project is NOT a "turn $5 into $1,000,000" challenge.

This is a quantitative finance research project designed to answer a serious question:

> How do institutional-quality quantitative research methodologies perform when subjected to extreme real-world capital constraints?

The system will begin with exactly $5.00 of deployable capital and will operate under realistic market conditions including:

* Transaction costs
* Bid/ask spreads
* Slippage
* Market impact
* Fractional share constraints
* Execution latency
* Broker limitations
* Tax implications
* Risk management requirements

The primary output of this project is not profit.

The primary output is:

1. Research
2. Analysis
3. Documentation
4. Statistical findings
5. Reproducible experiments
6. Open quantitative finance knowledge

The project should be developed as if it were a miniature quantitative hedge fund research platform.

---

# CORE RESEARCH QUESTIONS

The system should attempt to answer:

### Research Question 1

Can advanced quantitative techniques outperform simple benchmarks under severe capital constraints?

### Research Question 2

How much alpha survives real-world execution?

### Research Question 3

Which models degrade least when transaction costs dominate returns?

### Research Question 4

Can AI/ML systems produce statistically significant signals in a low-capital environment?

### Research Question 5

What is the relationship between:

* Model complexity
* Transaction frequency
* Capital efficiency
* Risk-adjusted returns

---

# NON-NEGOTIABLE PRINCIPLES

## Principle 1

No gambling.

No meme stocks.

No YOLO trades.

No social media sentiment chasing.

No "get rich quick" behavior.

---

## Principle 2

Research first.

Every strategy must survive:

* Backtesting
* Walk-forward testing
* Paper trading

before touching live capital.

---

## Principle 3

Everything is documented.

Every decision.

Every experiment.

Every failure.

Every trade.

Every model.

---

## Principle 4

Reproducibility.

Every result should be reproducible from source code.

---

# SYSTEM ARCHITECTURE

```text
five-dollar-quant/

├── data/
│
├── research/
│
├── models/
│
├── backtests/
│
├── execution/
│
├── dashboard/
│
├── reports/
│
├── experiments/
│
├── docs/
│
├── notebooks/
│
├── tests/
│
├── config/
│
└── infrastructure/
```

---

# DATA ENGINEERING LAYER

## Goal

Create institutional-quality datasets.

---

## Data Sources

Priority ranking:

### Tier 1

* Alpaca
* Polygon
* AlphaVantage
* Stooq
* Yahoo Finance

### Tier 2

* FRED
* Treasury Data
* SEC Filings

### Tier 3

Alternative data

Including:

* News
* Sentiment
* Search trends

---

## Stored Data

For every asset store:

### OHLCV

* Open
* High
* Low
* Close
* Volume

---

### Derived Data

* Log returns
* Realized volatility
* ATR
* Drawdowns
* Rolling Sharpe
* Rolling Sortino

---

### Macro Data

Store:

* VIX
* Interest rates
* Yield curve
* Inflation
* Employment data

---

# RESEARCH ENGINE

This is the heart of the project.

---

# STRATEGY CATEGORY 1

## Buy and Hold

Purpose:

Benchmark.

Strategies:

* SPY
* QQQ
* IWM

---

# STRATEGY CATEGORY 2

## Trend Following

Implement:

### Moving Average Crossover

Test:

* 10/20
* 20/50
* 50/200

---

### Donchian Breakouts

Implement:

* 20-day
* 50-day

---

### Volatility Filter

Trade only when:

```text
Volatility < threshold
```

---

# STRATEGY CATEGORY 3

## Mean Reversion

Implement:

### RSI

Threshold search:

* 20
* 25
* 30

---

### Bollinger Bands

Test:

* 1 std
* 2 std
* 3 std

---

### Z-Score Reversion

Rolling windows:

* 5
* 10
* 20

---

# STRATEGY CATEGORY 4

## Statistical Arbitrage Research

Even if not tradable with $5.

Research value remains high.

Implement:

* Cointegration testing
* Engle-Granger
* Johansen
* Pair selection

Examples:

* SPY / IVV
* KO / PEP
* XOM / CVX

---

# STRATEGY CATEGORY 5

## Factor Investing

Build factor scores.

Factors:

### Momentum

### Value

### Quality

### Volatility

### Profitability

### Growth

---

Rank securities.

Create composite scores.

---

# MACHINE LEARNING RESEARCH

# Objective

Predict probability distributions.

NOT raw prices.

---

# Models

Implement:

### Logistic Regression

### Random Forest

### XGBoost

### LightGBM

### CatBoost

### MLP

### Temporal CNN

### LSTM

### Transformer-based forecasting

---

# Target Variables

Predict:

### Binary Direction

```text
Up
Down
```

---

### Return Buckets

```text
Large Loss
Small Loss
Flat
Small Gain
Large Gain
```

---

### Regime Classification

```text
Bull
Bear
Sideways
Volatile
```

---

# FEATURE ENGINEERING

Build extensive features.

---

## Technical

* RSI
* MACD
* ATR
* OBV
* ADX

---

## Statistical

* Skewness
* Kurtosis
* Autocorrelation
* Hurst Exponent

---

## Volatility

* GARCH
* EWMA

---

## Market Breadth

* Advance/Decline
* Sector strength

---

## Macro

* Rates
* VIX
* Yield curve

---

## Alternative

Later:

* News embeddings
* Sentiment
* LLM-generated signals

---

# AI RESEARCH

## Goal

Use LLMs as research assistants.

Never direct traders.

---

Potential uses:

### Research Summaries

### Earnings Analysis

### SEC Filing Extraction

### Feature Generation

### Hypothesis Generation

### Trade Journal Analysis

---

# REGIME DETECTION SYSTEM

Critical component.

Implement:

### Hidden Markov Models

### Bayesian Regime Switching

### Volatility Clustering

### Change Point Detection

---

Output:

```text
Risk On
Risk Off
Crisis
Recovery
Trend
Mean Reversion
```

---

# BACKTESTING FRAMEWORK

Must support:

## Walk Forward Validation

## Expanding Window Validation

## Rolling Window Validation

## Monte Carlo Simulation

## Bootstrap Resampling

---

# COST MODELING

Required.

Most backtests fail here.

Include:

### Spread Costs

### Slippage

### Fractional Share Constraints

### Latency

### Order Rejections

---

# EXECUTION SYSTEM

Broker:

Alpaca

---

Order Types:

### Market

### Limit

### Fractional

---

Implement:

### Order Manager

### Position Manager

### Portfolio Manager

### Risk Engine

---

# RISK MANAGEMENT

Most important component.

---

## Hard Limits

Starting Capital:

```text
$5.00
```

---

Maximum Exposure:

```text
90%
```

---

Cash Buffer:

```text
10%
```

---

Maximum Drawdown Before Shutdown:

```text
20%
```

---

Daily Loss Limit:

```text
5%
```

---

Trade Frequency Limit:

```text
1 trade/day initially
```

---

# PERFORMANCE ANALYTICS

Track:

### CAGR

### Sharpe

### Sortino

### Calmar

### Max Drawdown

### Win Rate

### Profit Factor

### Alpha

### Beta

### Information Ratio

---

# DASHBOARD

Build institutional-style dashboard.

Pages:

---

## Overview

* Equity Curve
* Current Holdings
* Account Value

---

## Strategies

* Leaderboards
* Performance Metrics

---

## Research

* Experiments
* Model Results

---

## Live Trading

* Open Positions
* Orders
* Risk Metrics

---

## Journal

* Trade Notes
* AI Analysis
* Lessons Learned

---

# VISUALIZATION REQUIREMENTS

Create publication-quality graphics.

Include:

### Equity Curves

### Rolling Sharpe

### Drawdown Waterfalls

### Correlation Heatmaps

### Feature Importance

### Regime Timelines

### Prediction Confidence

### Trade Distribution

---

# DOCUMENTATION

Every experiment gets:

```text
Hypothesis

Data

Methodology

Results

Statistical Significance

Conclusion

Next Steps
```

---

# FINAL RESEARCH PAPER

Target:

40–100 pages

Sections:

1. Introduction
2. Literature Review
3. Data
4. Methodology
5. Models
6. Backtesting
7. Live Trading
8. Results
9. Limitations
10. Future Work

---

# SUCCESS CRITERIA

Success is NOT:

Making money.

Success IS:

Building a rigorous quantitative finance research platform that:

* Generates reproducible research
* Demonstrates statistical rigor
* Produces institutional-quality documentation
* Creates a public record of findings
* Shows honest evaluation of model performance
* Survives real-world deployment

If profitability occurs, treat it as a secondary outcome rather than the primary objective.
