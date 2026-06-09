# PROJECT HANDOFF — v2.0

# FIVE DOLLAR QUANT

## A Simulation-First Study of Institutional Quantitative Methods Under Extreme Capital Constraints

Author: Ajai Upadhyaya
Version: 2.0
Supersedes: v1.0 (live-capital variant)

---

# WHAT CHANGED FROM v1.0

v1.0 planned live deployment of $5.00 real capital. v2.0 removes live money entirely. The project is now:

1. **Historical simulation (primary):** 10 years of real market data, with a synthetic hard constraint of $5.00 starting capital and a fully realistic friction model.
2. **Paper trading (secondary, "live" element):** continuous forward deployment of surviving strategies on Alpaca's paper environment, configured to enforce the same $5.00 constraint.
3. **Research output (the actual deliverable):** experiment reports, capital-viability curves, statistical analysis, and a final paper.

Nothing about the rigor changes. The friction model must be at least as honest as a live account would have forced it to be — *more* honest, because simulation makes it easy to cheat.

---

# CENTRAL RESEARCH QUESTION (REFRAMED)

> **What is the minimum viable capital for each class of quantitative strategy, and how does a $5.00 account sit relative to that frontier — measured on 10 years of real data under realistic frictions?**

Supporting questions:

- **RQ1 — Alpha decay under friction:** How much of each strategy's gross alpha survives spreads, fee minimums, fractional-share constraints, and settlement rules at the $5 level? At $50? $500? $5,000? $50,000?
- **RQ2 — Cost dominance threshold:** At what capital level does each strategy's expected edge cross its expected friction (break-even capital)?
- **RQ3 — Complexity vs. robustness:** Do more complex models (ML) degrade faster or slower than simple rules when costs dominate?
- **RQ4 — Statistical power:** Given realized trade frequency, what effect sizes are even detectable, and over what horizon? (Pre-registered power analysis; deflated Sharpe as the standard.)
- **RQ5 — Sim-to-paper transfer:** Do strategies tuned on history hold up in forward paper trading, and how large is the live-vs-backtest performance gap?

---

# NON-NEGOTIABLE PRINCIPLES

1. **The $5 constraint is sacred in every simulation.** No experiment may silently assume more capital, ignore the $1 minimum order, or allow shorts on fractional positions. Capital-scaling experiments (RQ1/RQ2) must explicitly declare their capital level.
2. **Overfitting is the primary scientific risk.** "Highly refined and retuned on historical data" means *disciplined hyperparameter search inside a validation protocol* — never tuning on the test period. Every reported result carries a Deflated Sharpe Ratio and a Probability of Backtest Overfitting estimate. Tuning is unlimited only inside in-sample windows.
3. **The friction model is the project's crown jewel.** It is versioned, documented, unit-tested, and cited in every result. A result produced under an older friction model version must be flagged.
4. **Everything is reproducible.** One command rebuilds any figure or table from raw data + config + seed.
5. **Negative results are first-class results.** "Strategy X cannot overcome friction below $Y capital" is a finding, not a failure.
6. **Publish serially.** One public experiment report per completed experiment. The final paper assembles them; it does not gate them.

---

# DATA SPECIFICATION

## Universe (Phase 1 — keep it small)

~10 highly liquid US ETFs: SPY, QQQ, IWM, TLT, IEF, GLD, XLE, XLF, XLK, and one pair candidate set (e.g., XOM/CVX, KO/PEP for sim-only stat arb).

Rationale: liquid ETFs have the tightest spreads, making them the *best case* for a $5 account. If strategies fail here, they fail everywhere — a clean scientific result.

## History

- **Window:** 10 years (2016-06 through 2026-06), daily OHLCV. Adjusted and unadjusted prices both stored (adjusted for signals, unadjusted for execution simulation).
- **Reserved test set:** the final 2 years (2024-06 → 2026-06) are LOCKED. No model, parameter, or design decision may be informed by them. They are touched once, at the end, per strategy family.
- **Sources:** Alpaca / Stooq / Yahoo as primary OHLCV; FRED for VIX, rates, yield curve. Cross-validate at least two sources; log discrepancies.

## Derived data (computed once, cached, versioned)

Log returns, realized vol (multiple windows), ATR, drawdown series, rolling Sharpe/Sortino, and the full feature store for ML (see below).

---

# FRICTION MODEL (THE CORE ARTIFACT)

Every simulated order passes through a broker emulator that mirrors documented Alpaca retail behavior:

1. **Minimum notional:** $1.00 per order. With $5.00 and a 10% cash buffer, max 4 concurrent positions. Order sizes below $1 are rejected (and the rejection is logged — rejection rates are themselves a metric).
2. **Fractional fills at NBBO, no price improvement:** charge the full half-spread on entry and exit. Spread estimated per-asset from historical quote data where available, else from a documented spread model (e.g., SPY ≈ 1 tick; widen during high-VIX regimes).
3. **Long-only for fractional positions.** No shorting. Stat arb is therefore simulation-only at higher capital tiers, or implemented long-only (underweight/overweight vs. benchmark).
4. **Regulatory fee floor:** model SEC fee + FINRA TAF on sells *including per-trade minimums* (penny-level minimums are ~100bp on a $1 ticket — this is the cost-dominance mechanism made concrete). Calibrate against Alpaca's published fee pass-through; document assumptions.
5. **Cash account settlement (T+1):** sale proceeds are unavailable until the next business day. The simulator must enforce settled-cash-only buying and flag would-be good-faith violations. This naturally caps turnover and must not be bypassed.
6. **Latency / staleness:** signals computed on close of day T execute at the open of day T+1 (or next-day VWAP variant as a sensitivity check). No same-bar fills, ever.
7. **Slippage stress tier:** every headline result is re-run at 2x and 5x modeled friction as a robustness band.

**Capital-tier switch:** the same engine runs at $5 / $50 / $500 / $5,000 / $50,000, relaxing only the constraints that genuinely relax with capital (granularity, settlement binding less often), to produce the capital-viability curves.

---

# STRATEGY ROADMAP (DESCOPED, ORDERED)

## Tier 0 — Benchmarks (Week 1 of research)
- Buy-and-hold SPY, QQQ, IWM at $5 (one position) — the bar every strategy must clear *net of friction*.
- 60/40-style two-asset hold (SPY/TLT) at $5 — tests whether even rebalancing survives the $1 minimum.

## Tier 1 — Simple rules (Phase 1)
- **Trend:** MA crossover (10/20, 20/50, 50/200) with a volatility filter; Donchian 20/50.
- **Mean reversion:** rolling z-score (5/10/20 windows); RSI thresholds (20/25/30); Bollinger (1/2/3σ).
- All parameter grids are searched **only** via walk-forward inside the in-sample period.

## Tier 2 — Cross-sectional & regime (Phase 2)
- **Cointegration stat arb (sim-only / higher capital tiers):** Engle-Granger and Johansen on the pair set; report explicitly at which capital tier it becomes executable.
- **Regime layer:** Hidden Markov Model (2–3 states) on returns + realized vol, used as a *filter* on Tier 1 strategies (trade trend in trend regimes, reversion in chop). Regime detection is a meta-layer, not a standalone strategy.

## Tier 3 — Machine learning (Phase 3, gated)
- Exactly two models to start: **logistic regression** and **LightGBM**, predicting 5-bucket return classes and binary direction.
- Validation: purged k-fold with embargo (López de Prado), then walk-forward. Feature set: the technical/statistical/vol/macro features from v1.0, pruned by importance stability.
- Deep models (LSTM/Transformer/TCN) are admitted **only if** Tier 3 baselines show any out-of-sample signal worth scaling. Expectation: they won't on daily ETF bars; documenting that is the finding.

Cut from scope entirely (for now): CatBoost/XGBoost/RF redundancy, factor composite scoring across a broad universe, alternative data, news/LLM signals, market breadth. These return only if earlier phases produce something worth extending.

---

# VALIDATION & STATISTICS PROTOCOL

Mandatory for every reported strategy:

1. **Split discipline:** In-sample 2016–2022 (tune freely, walk-forward), validation 2022–2024 (model selection), locked test 2024–2026 (touched once).
2. **Walk-forward + expanding-window** as primary; rolling-window as sensitivity.
3. **Deflated Sharpe Ratio** accounting for the number of trials (log every configuration ever run — the trial counter is part of the experiment record).
4. **Probability of Backtest Overfitting (CSCV)** per strategy family.
5. **Monte Carlo / bootstrap** resampling of trade sequences for drawdown and ruin-probability distributions at $5 (with a $1 minimum ticket, "ruin" = capital < $1.11, the smallest viable round trip + fee floor — compute and report time-to-ruin distributions).
6. **Pre-registered power analysis:** before paper trading begins, publish the expected trade count and the minimum detectable Sharpe over 6/12/24 months. State plainly what the paper-trading phase can and cannot prove.

---

# PAPER TRADING ("LIVE" PHASE)

- **Platform:** Alpaca paper API. Paper accounts default to $100K equity — the harness must enforce a **virtual $5.00 ledger** on top: the system tracks its own capital, refuses orders the $5 account couldn't place, applies the friction model's fee floor synthetically (paper fills are friction-free, so frictions are charged in the ledger), and enforces T+1 settled-cash logic.
- **Promotion rule:** a strategy reaches paper trading only after surviving in-sample tuning, validation selection, the locked test, DSR > 0 at 95%, and PBO below a pre-declared threshold.
- **Duration:** minimum 6 months continuous, target 12.
- **Sim-to-paper audit (RQ5):** weekly automated comparison of paper fills vs. the simulator's predicted fills on identical signals; the divergence series is itself a published dataset.
- **Kill criteria:** virtual drawdown > 20% from peak, or daily loss > 5%, halts the strategy pending a written post-mortem.

---

# SYSTEM ARCHITECTURE

```text
five-dollar-quant/
├── config/          # YAML per experiment: universe, dates, capital tier, friction version, seed
├── data/            # raw/ (immutable), processed/ (cached, versioned)
├── frictions/       # broker emulator + fee schedule + tests  ← crown jewel
├── strategies/      # one module per strategy family, pure signal functions
├── backtest/        # engine: walk-forward, capital tiers, settlement, order lifecycle
├── validation/      # DSR, PBO/CSCV, bootstrap, power analysis
├── ml/              # feature store, purged CV, the two models
├── paper/           # Alpaca paper harness + virtual $5 ledger + sim-vs-paper auditor
├── experiments/     # numbered: EXP-001/, each with config + results + report.md
├── dashboard/       # equity curves, capital-viability curves, cost ledger, regime timeline
├── reports/         # public write-ups; final paper drafts
└── tests/           # friction model and engine correctness tests are non-optional
```

Engineering rules: this repo stays small and clean — separate from quant-trading. Pure-function signal code (no I/O in strategy modules). Every experiment is a config file, not a code branch. CI runs the test suite plus one smoke backtest.

---

# EXPERIMENT REPORT TEMPLATE

```text
EXP-NNN — Title
Hypothesis
Data & window (and friction model version)
Methodology (incl. full search space and trial count)
Results (net-of-friction, all capital tiers)
Statistical assessment (DSR, PBO, bootstrap bands)
Capital-viability finding (break-even capital estimate)
Limitations & threats to validity
Conclusion & next steps
```

---

# SIGNATURE VISUALS (BUILD THESE EARLY)

1. **Capital-viability frontier:** per strategy, net Sharpe (or CAGR) vs. starting capital on a log axis, with the break-even crossing marked. This is the project's flagship chart.
2. **Cost ledger:** cumulative friction (spread, fee floor, rejections) in cents and as % of capital for the $5 account, simulated and paper.
3. **Gross-vs-net alpha waterfall:** how each friction component eats the edge.
4. **Sim-to-paper divergence tracker.**
5. Standard set: equity curves with bootstrap bands, drawdown waterfalls, rolling DSR, regime timeline, feature importance stability.

---

# PHASED TIMELINE

**Phase 0 — Infrastructure & friction model (Weeks 1–3)**
Data layer for the 10-ETF universe; friction model v1 with full test suite; backtest engine with capital tiers and settlement logic; Tier 0 benchmarks run end-to-end. Deliverable: EXP-001 ("What does buy-and-hold cost at $5?") published.

**Phase 1 — Simple strategies (Weeks 4–9)**
Tier 1 trend + mean reversion, full validation protocol, first capital-viability curves. Deliverables: EXP-002…005; power analysis published.

**Phase 2 — Cross-sectional & regime (Weeks 10–14)**
Stat arb at higher capital tiers; HMM regime filter ablation (each Tier 1 strategy with/without regime gating). Deliverables: EXP-006…008.

**Phase 3 — ML (Months 4–5)**
Feature store, purged CV, logistic + LightGBM. Deliverable: EXP-009/010, including an honest "do simple ML models beat the rule-based survivors net of friction?" verdict.

**Phase 4 — Locked test + paper launch (Month 5–6)**
One-shot evaluation of selected strategies on 2024–2026 locked data; promote survivors to the paper harness; begin the 6–12 month forward run.

**Phase 5 — Continuous (Months 6–18)**
Paper monitoring, sim-vs-paper audits, quarterly interim reports, final paper (target 40–60 pages: Introduction, Literature, Data, Friction Model, Methodology, Results, Capital-Viability Analysis, Paper-Trading Results, Limitations, Future Work). Post to SSRN/arXiv q-fin; serialized versions on the portfolio site.

---

# SUCCESS CRITERIA

Success is NOT a profitable equity curve.

Success IS:
- A tested, versioned, citable friction model for micro-capital retail execution.
- Capital-viability curves for every strategy family, with uncertainty bands.
- At least 10 published experiment reports with full statistical hygiene (DSR, PBO, bootstrap).
- A pre-registered, completed 6+ month paper-trading run with a quantified sim-to-paper gap.
- A final paper whose limitations section is as rigorous as its results section.
- A codebase a stranger can clone and reproduce any figure from in one command.

If a strategy turns out to be net-positive at $5: treat it as a result to be stress-tested, not a victory to be celebrated.