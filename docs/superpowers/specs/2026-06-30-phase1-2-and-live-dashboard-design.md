# Design — Phase 1 + Phase 2 Research Engine & Live Interactive Dashboard

- **Project:** capitallimits / Five Dollar Quant (`fdq`)
- **Date:** 2026-06-30
- **Author:** Ajai Upadhyaya (with Claude)
- **Status:** Approved (spec all four milestones; execute + checkpoint milestone-by-milestone)
- **Supersedes/extends:** `updatedplan.md` v2.0 (this is the concrete build spec for Phases 1–2 plus the dashboard)

## 1. Goal

Advance `fdq` from Phase 0 (friction model, engine, Tier-0 benchmarks, EXP-001) to a
system that (a) runs a real **Phase 1 + Phase 2** research program — trend, mean-reversion,
walk-forward validation, HMM regime filtering, and cointegration stat-arb — and (b) exposes
all of it through a **dynamic, responsive, interactive web dashboard** that replaces the
static HTML tearsheet.

Every result remains **net-of-friction**, reproducible, and produced on **real cached market
data** only. Negative results are first-class. The central research question is unchanged:
*what is the minimum viable capital per strategy class, under realistic retail frictions?*

## 2. Non-negotiable constraints (inherited)

1. **No synthetic market data** in any backtest, experiment, or dashboard chart. Real cached
   parquet bars with valid `.meta.json` provenance only. (HMM/estimator unit tests may use
   controlled numeric sequences, never surfaced as market results.)
2. **The $5 constraint is sacred** in every simulation; capital-tier experiments declare tiers explicitly.
3. **Locked test set (2024-06 → 2026-06) is never touched** during tuning or dashboard exploration.
4. **Friction model is versioned and cited** in every result; results under an older version are flagged.
5. **Overfitting is the primary scientific risk:** every reported strategy carries DSR + PBO;
   tuning happens only inside in-sample walk-forward windows.
6. **Reproducible:** one command rebuilds any figure/table from raw data + config + seed.
7. Repo stays **small, clean, Python-only** (no Node build toolchain). Pure-function signal code.

## 3. Architecture overview

Two tracks share the existing backtest engine (`fdq/backtest/engine.py`) unchanged in its core
contract (dollar-notional positions, T+1 settlement, signal-at-close → fill-at-next-open).

```
fdq/
├── strategies/
│   ├── benchmarks.py      (exists)
│   ├── trend.py           NEW — MA crossover, Donchian, vol filter
│   ├── meanrev.py         NEW — z-score, RSI, Bollinger
│   ├── statarb.py         NEW — cointegration pairs (Phase 2)
│   └── regime_gated.py    NEW — RegimeGatedStrategy meta-wrapper (Phase 2)
├── regime/
│   └── hmm.py             NEW — in-repo Gaussian HMM (2–3 states)
├── validation/
│   ├── walkforward.py     NEW — walk-forward harness (Phase 1 crown)
│   ├── pbo.py             IMPLEMENT — CSCV PBO (currently a stub)
│   └── dsr.py, metrics.py, bootstrap.py (exist; wire trial_count)
├── experiment/
│   └── runner.py          EXTEND — support walk-forward + parameter-grid configs
├── dashboard/
│   ├── tearsheet.py       (exists; keep for static export)
│   ├── app.py             NEW — FastAPI application
│   ├── service.py         NEW — bar cache + on-demand backtest/sweep façade
│   ├── figures.py         NEW — Plotly figure builders (return JSON)
│   └── templates/, static/  NEW — Jinja + HTMX + responsive CSS
└── cli.py                 EXTEND — `fdq dashboard serve`, walk-forward experiment support
```

### Decisions (locked)
- **Dashboard stack:** FastAPI + server-rendered Jinja + **HTMX** (fragment swaps, no SPA) +
  **Plotly** via CDN for interactive charts. Responsive CSS grid; mobile-friendly.
- **HMM:** implemented in-repo (numpy/scipy EM), not `hmmlearn` — a tested, citable research artifact.
- **New dependencies:** `fastapi`, `uvicorn[standard]`, `python-multipart`, `plotly`, `statsmodels`.
  (`statsmodels` powers Engle-Granger/Johansen cointegration tests.)

## 4. Track A — Research engine (Phase 1 + Phase 2)

### 4.1 Tier-1 strategies (pure signal functions, long/flat, single-ETF)

All produce `target_weights` compatible with the existing engine: weight `1.0` when in-position,
absent/`0.0` when flat; `should_rebalance` returns `True` only when the signal changes state.
Each strategy declares its full parameter grid for walk-forward search.

- **`trend.py`**
  - `MACrossover(fast, slow)` — grids: (10,20), (20,50), (50,200).
  - `Donchian(window)` — 20, 50; enter on N-day high breakout, exit on N-day low.
  - Optional **volatility filter**: suppress entries when realized vol > threshold.
- **`meanrev.py`**
  - `ZScoreReversion(window, entry_z, exit_z)` — windows 5/10/20.
  - `RSIReversion(period, low, high)` — thresholds 20/25/30.
  - `Bollinger(window, n_std)` — 1/2/3σ.

Registered in `strategies.build_strategy`. Unit-tested for signal correctness on real fixtures.

### 4.2 Walk-forward validation harness (`validation/walkforward.py`)

- Splits the **in-sample** window (2016–2022) into sequential train→test folds
  (expanding-window primary; rolling-window as a sensitivity switch).
- For each fold: grid-search the strategy's parameters **on the train slice only**, pick the
  best by a configurable objective (default: net-of-friction Sharpe), then run the chosen
  params on the **immediately following OOS test slice**. No test slice ever informs its own fit.
- Stitches per-fold OOS equity into a single OOS curve; returns per-fold chosen params, the
  stitched curve, and the **total trial count** (every configuration evaluated) for DSR.
- **Leakage guard is explicit and unit-tested:** an assertion that a fold's test dates are
  strictly after its train dates, and that no grid evaluation reads test-slice data.

Wire the resulting trial count into `deflated_sharpe`; the naive `trial_sharpes=[0.0]` placeholder
in `runner._summarize_strategy` is replaced with the real per-configuration Sharpe vector.

### 4.3 PBO / CSCV (`validation/pbo.py`)

Replace the `NotImplementedError` stub with a real **Combinatorially-Symmetric Cross-Validation**
estimator (López de Prado): partition the trial performance matrix into S combinatorial
train/test splits, compute the fraction of splits where the in-sample-best configuration
underperforms the test-set median → **Probability of Backtest Overfitting**. Reported per
strategy family. Unit-tested against constructed matrices with known PBO (e.g., pure-noise
trials → PBO ≈ 0.5; a genuinely dominant configuration → low PBO).

### 4.4 Regime layer (`regime/hmm.py` + `strategies/regime_gated.py`) — Phase 2

- **Gaussian HMM**, 2–3 hidden states, fit via EM (Baum-Welch) on features = {daily log return,
  rolling realized vol}. numpy/scipy only. Deterministic given seed. Returns per-day state
  posteriors and a Viterbi path.
- **`RegimeGatedStrategy`**: wraps a Tier-1 strategy and a regime map (e.g., "trade the trend
  strategy only in trending states; go flat / switch to reversion in chop"). It is a
  **meta-filter, not a standalone strategy**.
- **Ablation:** each Tier-1 survivor run with and without the gate; the delta is the finding.
- HMM tests use controlled numeric sequences with known regime structure (not market data).

### 4.5 Cointegration stat-arb (`strategies/statarb.py`) — Phase 2

- Pairs: **XOM/CVX, KO/PEP, SPY/IVV**. Tests: **Engle-Granger** (`statsmodels` ADF on the
  spread residual) and **Johansen** (`statsmodels` `coint_johansen`). Hedge ratio from OLS /
  Johansen eigenvector, estimated on train slices only.
- **Long-only realization** (overweight/underweight the two legs relative to a benchmark), since
  the friction model forbids shorting fractional positions. Reported **sim-only at higher capital
  tiers**, and the spec explicitly states **the capital tier at which the pair becomes executable**
  under the $1 minimum + settlement rules.

### 4.6 Experiments (each: config → backtest/walk-forward → summary.json + report.md + tearsheet)

- **EXP-002 — Trend under friction:** capital-viability curves for MA-crossover + Donchian,
  walk-forward, DSR + PBO, 1×/2×/5× friction stress.
- **EXP-003 — Mean-reversion under friction:** z-score/RSI/Bollinger, same protocol.
- **EXP-004 — Does regime gating help?:** HMM-gated vs ungated ablation across Tier-1 survivors.
- **EXP-005 — Cointegration stat-arb capital frontier:** pair tests + executability-by-tier finding.

The `runner` is extended so an experiment config can declare `mode: walkforward` with a parameter
grid, in addition to the existing single-run `mode: fixed`.

## 5. Track B — Live interactive dashboard

### 5.1 Service layer (`dashboard/service.py`)
- Loads validated cached bars + macro **once** into an in-memory cache (fast repeat runs).
- `run_backtest_spec(strategy, params, tier, stress)` → serializable result (equity, returns,
  metrics, cost ledger, trades, rejections).
- `capital_sweep_spec(...)` → per-tier results for the viability frontier.
- Reuses `engine` + `runner` internals; **enforces the same friction model and split discipline**
  (the `/explore` UI cannot select locked-test dates).

### 5.2 FastAPI app (`dashboard/app.py`) + templates
- **`GET /`** — overview: list of completed experiments, headline equity curves, "cost of $5" callouts.
- **`GET /experiment/{id}`** — detail: capital-viability frontier, cost-ledger waterfall,
  gross-vs-net alpha, trades table, DSR/PBO badges, rejection stats.
- **`GET /explore`** — interactive lab: form controls (strategy, params, capital tier,
  friction-stress multiplier). On change, **HTMX POST → `POST /api/backtest`** runs a live
  backtest and swaps in updated Plotly charts + a metrics table — no full page reload.
- **`POST /api/backtest`** — runs the spec via the service layer, returns an HTML fragment
  (Plotly figure JSON embedded + metrics). Guardrail: rejects locked-test date ranges.
- **`GET /healthz`** — liveness for tests/ops.

### 5.3 Figures (`dashboard/figures.py`)
Plotly builders returning JSON: capital-viability frontier (log-x), cost-ledger waterfall,
gross-vs-net alpha waterfall, equity curve with bootstrap bands, drawdown waterfall,
regime timeline (colored HMM states under the price). Shared by dashboard and static tearsheet.

### 5.4 Responsiveness & aesthetic
- HTMX for partial swaps; Plotly responsive config (`responsive: true`).
- CSS grid that reflows to single-column on narrow screens; restrained institutional look
  (no gratuitous gradients/animations) consistent with the user's design preferences.
- **`fdq dashboard serve --port 8080 [--host]`** launches uvicorn.

## 6. Testing strategy

- **Strategy signals:** deterministic correctness on real fixture slices (entry/exit timing).
- **Walk-forward:** no-leakage assertions (test dates strictly post-train; grid never reads test);
  reproducibility given seed.
- **PBO/CSCV:** known-answer matrices (noise → ~0.5; dominant config → low).
- **HMM:** state recovery + determinism on controlled numeric sequences.
- **Stat-arb:** cointegration detection on a known-cointegrated constructed series; hedge-ratio sanity.
- **Dashboard:** FastAPI `TestClient` — every route returns 200; `/api/backtest` runs a real
  fixture-backed backtest and returns metrics; locked-test guardrail returns an error.
- **CI:** existing smoke backtest retained; ruff + mypy strict + pytest all green (CI adds the
  new deps). Dashboard route tests run headless (no browser needed).

## 7. Build order (checkpoint after each milestone)

1. **Milestone 1 — Phase 1 core:** trend + mean-reversion strategies, walk-forward harness,
   PBO implementation, DSR trial-count wiring, runner walk-forward mode. → checkpoint.
2. **Milestone 2 — Phase 1 experiments:** EXP-002 (trend) + EXP-003 (mean-reversion) with
   reports/tearsheets/summaries. → checkpoint.
3. **Milestone 3 — Phase 2 research:** HMM regime + RegimeGatedStrategy + stat-arb; EXP-004
   (regime ablation) + EXP-005 (stat-arb frontier). → checkpoint.
4. **Milestone 4 — Live dashboard:** service layer, FastAPI app, figures, templates, `serve` CLI,
   route tests. → checkpoint.

Each milestone: TDD where practical, ruff+mypy+pytest green, then a written checkpoint summary
before proceeding.

## 8. Success criteria

- Four new experiment reports (EXP-002…005) with full statistical hygiene (net-of-friction,
  DSR, PBO, bootstrap bands, capital-viability curves), honest about negative results.
- A working, responsive dashboard: browse experiments and **live-explore** strategy/params/
  capital/friction with sub-second re-runs, all on real data, locked test protected.
- Repo stays clean: ruff + mypy strict + full pytest suite green; one-command reproducibility
  preserved; no synthetic market data anywhere.

## 9. Out of scope (this spec)

Tier-3 ML (logistic/LightGBM), paper-trading harness + virtual-$5 ledger, deep models, alt-data,
deployment/hosting of the dashboard. These remain future phases per `updatedplan.md`.
