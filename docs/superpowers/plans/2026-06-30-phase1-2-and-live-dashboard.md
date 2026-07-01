# Phase 1 + 2 Research Engine & Live Dashboard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `fdq` from Phase 0 (benchmarks) into a Phase 1+2 research engine (trend, mean-reversion, walk-forward, HMM regime, cointegration stat-arb) exposed through a dynamic, responsive FastAPI + HTMX + Plotly dashboard.

**Architecture:** New pure-function strategy modules and a walk-forward validation harness plug into the existing dollar-notional backtest engine unchanged. A regime layer and stat-arb module extend the research surface. A FastAPI service wraps the engine for live, sub-second, on-demand backtests in the browser. All results are net-of-friction on real cached data.

**Tech Stack:** Python 3.12, uv, pandas/numpy/scipy, statsmodels (stat-arb), FastAPI + uvicorn + HTMX + Plotly (dashboard). Existing: click, rich, pydantic, matplotlib, jinja2.

## Global Constraints

Copied verbatim from the spec — every task's requirements implicitly include these:

- **No synthetic market data** in any backtest, experiment, or dashboard chart — real cached parquet bars with valid `.meta.json` provenance only. Estimator unit tests (HMM/PBO) may use controlled numeric sequences, never surfaced as market results.
- **Locked test set 2024-06-01 → 2026-06-01 is never touched** during tuning or dashboard exploration.
- **The $5 constraint is sacred**; capital-tier experiments declare tiers explicitly. Capital tiers: `[5, 50, 500, 5000, 50000]`.
- **Friction model is versioned (`1.0.0`) and cited** in every result.
- **Every reported strategy carries DSR + PBO**; tuning happens only inside in-sample walk-forward windows. In-sample 2016-06-01 → 2022-05-31; validation 2022-06-01 → 2024-05-31.
- **Reproducible** from raw data + config + seed (default seed `42`).
- **Repo stays small, clean, Python-only** — no Node toolchain. Pure-function signal code (no I/O in strategy modules).
- **Tooling gates per task:** `uv run ruff check fdq tests`, `uv run mypy fdq` (strict), `uv run pytest` must all pass before commit. Line length 100. `from __future__ import annotations` at top of every module.

## Existing interfaces this plan builds on (verbatim from the codebase)

- `fdq.strategies.base.Strategy` (ABC): `__init__(self, params: dict[str, object] | None = None)`; abstract `target_weights(self, asof: date, bars: pd.DataFrame) -> pd.Series`; `should_rebalance(self, asof, bars) -> bool` (default `True`). `spec: StrategySpec(slug, name, description, universe)`.
- `bars` is a **MultiIndex-column** DataFrame: columns are `(symbol, field)` where field ∈ {`open`,`high`,`low`,`close`,`close_adj`,`volume`}; index is a `DatetimeIndex`. Access a series with `bars[(symbol, field)]`.
- `fdq.backtest.engine.run_backtest(strategy, bars, config, macro=None, start=None, end=None) -> BacktestResult` with `.equity_curve`, `.returns`, `.trades`, `.cost_ledger`, `.metadata`. `BacktestConfig(starting_capital=5.0, friction=None, execution="next_open", seed=42)`.
- `fdq.backtest.engine.run_capital_sweep(make_strategy, bars, tiers, friction, macro=None, start=None, end=None, seed=42) -> dict[float, BacktestResult]`.
- `fdq.frictions.config.FrictionConfig` + `load_friction_config(path="config/friction_v1.yaml", stress_multiplier=1.0)`.
- `fdq.validation.metrics`: `sharpe(returns, annualize=True)`, `cagr`, `max_drawdown`, `sortino`, `total_return`, `turnover(trades, equity)`.
- `fdq.validation.dsr.deflated_sharpe(returns: pd.Series, trial_sharpes: np.ndarray) -> float`; `probabilistic_sharpe(returns, sr_benchmark)`.
- `fdq.data.ensure.load_validated_bars(BarRequest(symbols, start, end)) -> pd.DataFrame`; `load_validated_macro() -> pd.DataFrame`.
- `fdq.strategies.benchmarks.build_strategy(name: str, params: dict) -> Strategy`.
- `fdq.experiment.runner.run_experiment(config_path: Path, tearsheet=True, report=True, data_dir=None) -> Path`.

---

# MILESTONE 1 — Phase 1 core (strategies + walk-forward + PBO + DSR wiring)

Fully detailed below. Checkpoint after Task 1.8.

### Task 1.1: Shared signal helpers

**Files:**
- Create: `fdq/strategies/signals.py`
- Test: `tests/test_signals.py`

**Interfaces:**
- Produces:
  - `price_series(bars: pd.DataFrame, symbol: str, field: str = "close_adj", asof: date | None = None) -> pd.Series` — returns the symbol's price series (falls back to `close` if `close_adj` absent), truncated to `<= asof` when given.
  - `sma(s: pd.Series, window: int) -> pd.Series`
  - `rolling_zscore(s: pd.Series, window: int) -> pd.Series`
  - `rsi(s: pd.Series, period: int) -> pd.Series`
  - `realized_vol(close: pd.Series, window: int = 21) -> pd.Series` — annualized (×√252) rolling std of log returns.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_signals.py
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from fdq.strategies.signals import price_series, realized_vol, rolling_zscore, rsi, sma


def _bars() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    close = pd.Series(np.linspace(100, 160, 60), index=idx)
    frame = pd.DataFrame({"close": close, "close_adj": close, "high": close + 1,
                          "low": close - 1, "open": close, "volume": 1e6})
    return pd.concat({"SPY": frame}, axis=1)


def test_price_series_prefers_adjusted_and_truncates() -> None:
    bars = _bars()
    s = price_series(bars, "SPY", asof=date(2020, 1, 15))
    assert s.index.max().date() <= date(2020, 1, 15)
    assert s.iloc[0] == 100.0


def test_sma_matches_rolling_mean() -> None:
    bars = _bars()
    s = price_series(bars, "SPY")
    assert abs(sma(s, 5).iloc[-1] - s.iloc[-5:].mean()) < 1e-9


def test_rsi_bounds_and_zscore_and_vol() -> None:
    bars = _bars()
    s = price_series(bars, "SPY")
    r = rsi(s, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    assert abs(rolling_zscore(s, 10).iloc[-1]) < 5
    assert realized_vol(s, 21).dropna().iloc[-1] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_signals.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'fdq.strategies.signals'`

- [ ] **Step 3: Write minimal implementation**

```python
# fdq/strategies/signals.py
"""Pure signal helpers shared across strategy modules. No I/O."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def price_series(
    bars: pd.DataFrame, symbol: str, field: str = "close_adj", asof: date | None = None
) -> pd.Series:
    col = (symbol, field)
    if col not in bars.columns:
        col = (symbol, "close")
    s = pd.Series(bars[col], dtype=float).dropna()
    if asof is not None:
        s = s.loc[s.index <= pd.Timestamp(asof)]
    return s


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).mean()


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window).mean()
    std = s.rolling(window).std(ddof=1)
    return (s - mean) / std.replace(0.0, np.nan)


def rsi(s: pd.Series, period: int) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0.0).rolling(period).mean()
    loss = (-delta.clip(upper=0.0)).rolling(period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def realized_vol(close: pd.Series, window: int = 21) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std(ddof=1) * np.sqrt(252)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_signals.py -q && uv run ruff check fdq tests && uv run mypy fdq`
Expected: PASS, no lint/type errors.

- [ ] **Step 5: Commit**

```bash
git add fdq/strategies/signals.py tests/test_signals.py
git commit -m "feat(strategies): shared pure signal helpers"
```

---

### Task 1.2: Trend strategies (MA crossover, Donchian, vol filter)

**Files:**
- Create: `fdq/strategies/trend.py`
- Test: `tests/test_trend.py`

**Interfaces:**
- Consumes: `Strategy`, `StrategySpec` (base); `price_series`, `sma`, `realized_vol` (Task 1.1).
- Produces:
  - `MACrossover(params)` — params: `symbol` (str, default "SPY"), `fast` (int), `slow` (int), optional `vol_max` (float | None annualized-vol entry ceiling).
  - `Donchian(params)` — params: `symbol`, `window` (int), optional `vol_max`.
  - Both are long/flat single-ETF; `target_weights` returns `pd.Series({symbol: 1.0})` when in-position else empty `pd.Series(dtype=float)`. `should_rebalance` returns True only when desired state differs from current held state.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trend.py
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from fdq.strategies.trend import Donchian, MACrossover


def _trending_bars() -> pd.DataFrame:
    idx = pd.date_range("2019-01-01", periods=300, freq="B")
    close = pd.Series(100 * (1.0005 ** np.arange(300)), index=idx)
    frame = pd.DataFrame({"close": close, "close_adj": close, "high": close + 1,
                          "low": close - 1, "open": close, "volume": 1e6})
    return pd.concat({"SPY": frame}, axis=1)


def test_ma_crossover_long_in_uptrend() -> None:
    bars = _trending_bars()
    strat = MACrossover({"symbol": "SPY", "fast": 20, "slow": 50})
    asof = bars.index[-1].date()
    strat.should_rebalance(asof, bars)
    w = strat.target_weights(asof, bars)
    assert w.get("SPY", 0.0) == 1.0


def test_ma_crossover_flat_without_enough_history() -> None:
    bars = _trending_bars()
    strat = MACrossover({"symbol": "SPY", "fast": 20, "slow": 50})
    asof = bars.index[10].date()
    strat.should_rebalance(asof, bars)
    w = strat.target_weights(asof, bars)
    assert w.empty or w.get("SPY", 0.0) == 0.0


def test_vol_filter_blocks_entry() -> None:
    bars = _trending_bars()
    strat = MACrossover({"symbol": "SPY", "fast": 20, "slow": 50, "vol_max": 0.0001})
    asof = bars.index[-1].date()
    strat.should_rebalance(asof, bars)
    assert strat.target_weights(asof, bars).get("SPY", 0.0) == 0.0


def test_donchian_enters_on_breakout() -> None:
    bars = _trending_bars()
    strat = Donchian({"symbol": "SPY", "window": 20})
    asof = bars.index[-1].date()
    strat.should_rebalance(asof, bars)
    assert strat.target_weights(asof, bars).get("SPY", 0.0) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_trend.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'fdq.strategies.trend'`

- [ ] **Step 3: Write minimal implementation**

```python
# fdq/strategies/trend.py
"""Tier-1 trend strategies — long/flat, single ETF. Pure signal logic."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from fdq.strategies.base import Strategy, StrategySpec
from fdq.strategies.signals import price_series, realized_vol, sma


class _LongFlat(Strategy):
    """Shared long/flat state machine keyed on a boolean desired state."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.symbol = str(self.params.get("symbol", "SPY"))
        self.vol_max = self.params.get("vol_max")
        self._current = False
        self._pending = False
        self.spec = StrategySpec(self.slug(), self.name(), self.name(), [self.symbol])

    def slug(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def name(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def _desired(self, asof: date, bars: pd.DataFrame) -> bool:
        raise NotImplementedError

    def _vol_ok(self, asof: date, bars: pd.DataFrame) -> bool:
        if self.vol_max is None:
            return True
        close = price_series(bars, self.symbol, asof=asof)
        vol = realized_vol(close, 21).dropna()
        if vol.empty:
            return False
        return bool(vol.iloc[-1] <= float(self.vol_max))

    def should_rebalance(self, asof: date, bars: pd.DataFrame) -> bool:
        desired = self._desired(asof, bars)
        if desired and not self._vol_ok(asof, bars):
            desired = False
        if desired != self._current:
            self._pending = desired
            self._current = desired
            return True
        self._pending = self._current
        return False

    def target_weights(self, asof: date, bars: pd.DataFrame) -> pd.Series:
        if self._pending:
            return pd.Series({self.symbol: 1.0})
        return pd.Series(dtype=float)


class MACrossover(_LongFlat):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        p = params or {}
        self.fast = int(p.get("fast", 20))
        self.slow = int(p.get("slow", 50))
        super().__init__(params)

    def slug(self) -> str:
        return "ma_crossover"

    def name(self) -> str:
        return f"MA {self.fast}/{self.slow} {self.symbol}"

    def _desired(self, asof: date, bars: pd.DataFrame) -> bool:
        close = price_series(bars, self.symbol, asof=asof)
        if len(close) < self.slow + 1:
            return False
        fast_ma = sma(close, self.fast).iloc[-1]
        slow_ma = sma(close, self.slow).iloc[-1]
        return bool(fast_ma > slow_ma)


class Donchian(_LongFlat):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        p = params or {}
        self.window = int(p.get("window", 20))
        super().__init__(params)

    def slug(self) -> str:
        return "donchian"

    def name(self) -> str:
        return f"Donchian {self.window} {self.symbol}"

    def _desired(self, asof: date, bars: pd.DataFrame) -> bool:
        high = price_series(bars, self.symbol, field="high", asof=asof)
        low = price_series(bars, self.symbol, field="low", asof=asof)
        close = price_series(bars, self.symbol, asof=asof)
        if len(close) < self.window + 1:
            return self._current
        upper = high.iloc[-self.window - 1 : -1].max()
        lower = low.iloc[-self.window - 1 : -1].min()
        px = close.iloc[-1]
        if px >= upper:
            return True
        if px <= lower:
            return False
        return self._current
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_trend.py -q && uv run ruff check fdq tests && uv run mypy fdq`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fdq/strategies/trend.py tests/test_trend.py
git commit -m "feat(strategies): trend (MA crossover, Donchian) with vol filter"
```

---

### Task 1.3: Mean-reversion strategies (z-score, RSI, Bollinger)

**Files:**
- Create: `fdq/strategies/meanrev.py`
- Test: `tests/test_meanrev.py`

**Interfaces:**
- Consumes: `_LongFlat` (Task 1.2), `price_series`, `rolling_zscore`, `rsi`, `sma` (Task 1.1).
- Produces:
  - `ZScoreReversion(params)` — `symbol`, `window` (int), `entry_z` (float), `exit_z` (float, default 0.0).
  - `RSIReversion(params)` — `symbol`, `period` (int), `low` (float), `high` (float).
  - `Bollinger(params)` — `symbol`, `window` (int), `n_std` (float).
  - Same long/flat contract as Task 1.2.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_meanrev.py
from __future__ import annotations

import numpy as np
import pandas as pd

from fdq.strategies.meanrev import Bollinger, RSIReversion, ZScoreReversion


def _dip_bars() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    base = np.full(120, 100.0)
    base[-1] = 80.0  # sharp oversold dip on the last bar
    close = pd.Series(base, index=idx)
    frame = pd.DataFrame({"close": close, "close_adj": close, "high": close + 1,
                          "low": close - 1, "open": close, "volume": 1e6})
    return pd.concat({"SPY": frame}, axis=1)


def test_zscore_enters_long_on_dip() -> None:
    bars = _dip_bars()
    strat = ZScoreReversion({"symbol": "SPY", "window": 20, "entry_z": 1.5})
    asof = bars.index[-1].date()
    strat.should_rebalance(asof, bars)
    assert strat.target_weights(asof, bars).get("SPY", 0.0) == 1.0


def test_rsi_enters_long_when_oversold() -> None:
    bars = _dip_bars()
    strat = RSIReversion({"symbol": "SPY", "period": 14, "low": 30.0, "high": 70.0})
    asof = bars.index[-1].date()
    strat.should_rebalance(asof, bars)
    assert strat.target_weights(asof, bars).get("SPY", 0.0) == 1.0


def test_bollinger_enters_below_lower_band() -> None:
    bars = _dip_bars()
    strat = Bollinger({"symbol": "SPY", "window": 20, "n_std": 2.0})
    asof = bars.index[-1].date()
    strat.should_rebalance(asof, bars)
    assert strat.target_weights(asof, bars).get("SPY", 0.0) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_meanrev.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'fdq.strategies.meanrev'`

- [ ] **Step 3: Write minimal implementation**

```python
# fdq/strategies/meanrev.py
"""Tier-1 mean-reversion strategies — long/flat, single ETF. Pure signal logic."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from fdq.strategies.signals import price_series, rolling_zscore, rsi, sma
from fdq.strategies.trend import _LongFlat


class ZScoreReversion(_LongFlat):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        p = params or {}
        self.window = int(p.get("window", 20))
        self.entry_z = float(p.get("entry_z", 1.5))
        self.exit_z = float(p.get("exit_z", 0.0))
        super().__init__(params)

    def slug(self) -> str:
        return "zscore_reversion"

    def name(self) -> str:
        return f"Z-Reversion {self.window} {self.symbol}"

    def _desired(self, asof: date, bars: pd.DataFrame) -> bool:
        close = price_series(bars, self.symbol, asof=asof)
        if len(close) < self.window + 1:
            return self._current
        z = rolling_zscore(close, self.window).iloc[-1]
        if pd.isna(z):
            return self._current
        if z <= -self.entry_z:
            return True
        if z >= self.exit_z:
            return False
        return self._current


class RSIReversion(_LongFlat):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        p = params or {}
        self.period = int(p.get("period", 14))
        self.low = float(p.get("low", 30.0))
        self.high = float(p.get("high", 70.0))
        super().__init__(params)

    def slug(self) -> str:
        return "rsi_reversion"

    def name(self) -> str:
        return f"RSI {self.period} {self.symbol}"

    def _desired(self, asof: date, bars: pd.DataFrame) -> bool:
        close = price_series(bars, self.symbol, asof=asof)
        if len(close) < self.period + 1:
            return self._current
        val = rsi(close, self.period).iloc[-1]
        if pd.isna(val):
            return self._current
        if val <= self.low:
            return True
        if val >= self.high:
            return False
        return self._current


class Bollinger(_LongFlat):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        p = params or {}
        self.window = int(p.get("window", 20))
        self.n_std = float(p.get("n_std", 2.0))
        super().__init__(params)

    def slug(self) -> str:
        return "bollinger"

    def name(self) -> str:
        return f"Bollinger {self.window}/{self.n_std} {self.symbol}"

    def _desired(self, asof: date, bars: pd.DataFrame) -> bool:
        close = price_series(bars, self.symbol, asof=asof)
        if len(close) < self.window + 1:
            return self._current
        mid = sma(close, self.window).iloc[-1]
        std = close.rolling(self.window).std(ddof=1).iloc[-1]
        px = close.iloc[-1]
        if pd.isna(mid) or pd.isna(std):
            return self._current
        if px <= mid - self.n_std * std:
            return True
        if px >= mid:
            return False
        return self._current
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_meanrev.py -q && uv run ruff check fdq tests && uv run mypy fdq`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fdq/strategies/meanrev.py tests/test_meanrev.py
git commit -m "feat(strategies): mean-reversion (z-score, RSI, Bollinger)"
```

---

### Task 1.4: Register Tier-1 strategies in the factory

**Files:**
- Modify: `fdq/strategies/benchmarks.py` (extend `build_strategy`)
- Test: `tests/test_strategy_factory.py`

**Interfaces:**
- Consumes: all strategy classes above.
- Produces: `build_strategy(name, params)` resolves `"ma_crossover"`, `"donchian"`, `"zscore_reversion"`, `"rsi_reversion"`, `"bollinger"` in addition to the existing benchmarks.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strategy_factory.py
from __future__ import annotations

import pytest

from fdq.strategies.benchmarks import build_strategy


@pytest.mark.parametrize(
    "name,params",
    [
        ("ma_crossover", {"symbol": "SPY", "fast": 20, "slow": 50}),
        ("donchian", {"symbol": "SPY", "window": 20}),
        ("zscore_reversion", {"symbol": "SPY", "window": 20, "entry_z": 1.5}),
        ("rsi_reversion", {"symbol": "SPY", "period": 14, "low": 30, "high": 70}),
        ("bollinger", {"symbol": "SPY", "window": 20, "n_std": 2.0}),
    ],
)
def test_factory_builds_tier1(name: str, params: dict) -> None:
    strat = build_strategy(name, params)
    assert strat.spec.slug == name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_strategy_factory.py -q`
Expected: FAIL — `ValueError: Unknown strategy: ma_crossover`

- [ ] **Step 3: Write minimal implementation**

Replace the `build_strategy` function at the bottom of `fdq/strategies/benchmarks.py` with:

```python
def build_strategy(name: str, params: dict[str, Any]) -> Strategy:
    from fdq.strategies.meanrev import Bollinger, RSIReversion, ZScoreReversion
    from fdq.strategies.trend import Donchian, MACrossover

    registry: dict[str, type[Strategy]] = {
        "buy_and_hold": BuyAndHold,
        "balanced_6040": Balanced6040,
        "ma_crossover": MACrossover,
        "donchian": Donchian,
        "zscore_reversion": ZScoreReversion,
        "rsi_reversion": RSIReversion,
        "bollinger": Bollinger,
    }
    if name not in registry:
        msg = f"Unknown strategy: {name}"
        raise ValueError(msg)
    return registry[name](params)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_strategy_factory.py tests/test_engine.py -q && uv run ruff check fdq tests && uv run mypy fdq`
Expected: PASS (existing engine tests still green).

- [ ] **Step 5: Commit**

```bash
git add fdq/strategies/benchmarks.py tests/test_strategy_factory.py
git commit -m "feat(strategies): register Tier-1 strategies in factory"
```

---

### Task 1.5: Walk-forward validation harness

**Files:**
- Create: `fdq/validation/walkforward.py`
- Test: `tests/test_walkforward.py`

**Interfaces:**
- Consumes: `run_backtest`, `BacktestConfig` (engine); `FrictionConfig`; `sharpe` (metrics); `build_strategy`.
- Produces:
  - `@dataclass(frozen=True) Fold(train_start: date, train_end: date, test_start: date, test_end: date)`
  - `make_folds(index: pd.DatetimeIndex, n_folds: int, scheme: str = "expanding") -> list[Fold]`
  - `@dataclass WalkForwardResult(oos_returns: pd.Series, oos_equity: pd.Series, fold_params: list[dict[str, Any]], trial_sharpes: np.ndarray, n_trials: int)`
  - `grid_combos(grid: dict[str, list[Any]]) -> list[dict[str, Any]]`
  - `walk_forward(strategy_name: str, base_params: dict[str, Any], grid: dict[str, list[Any]], bars, tier: float, friction: FrictionConfig, macro, n_folds: int = 4, scheme: str = "expanding", seed: int = 42) -> WalkForwardResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_walkforward.py
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fdq.data.provenance import validate_provenance
from fdq.frictions.config import FrictionConfig
from fdq.validation.walkforward import Fold, grid_combos, make_folds, walk_forward

FIX = Path(__file__).parent / "fixtures" / "SPY.parquet"


def test_folds_have_no_leakage() -> None:
    idx = pd.date_range("2016-06-01", "2022-05-31", freq="B")
    folds = make_folds(idx, n_folds=4, scheme="expanding")
    assert len(folds) == 4
    for f in folds:
        assert f.test_start > f.train_end  # test strictly after train


def test_grid_combos_cartesian() -> None:
    combos = grid_combos({"fast": [10, 20], "slow": [50]})
    assert {"fast": 10, "slow": 50} in combos and len(combos) == 2


def test_walk_forward_runs_on_real_fixture() -> None:
    validate_provenance(FIX)
    spy = pd.read_parquet(FIX)
    bars = pd.concat({"SPY": spy}, axis=1)
    start = bars.index.min().date()
    end = bars.index.max().date()
    if (end - start).days < 400:
        pytest.skip("fixture too short for walk-forward")
    res = walk_forward(
        "ma_crossover",
        {"symbol": "SPY"},
        {"fast": [10, 20], "slow": [50]},
        bars,
        tier=5.0,
        friction=FrictionConfig(),
        macro=None,
        n_folds=3,
    )
    assert len(res.fold_params) == 3
    assert res.n_trials == res.trial_sharpes.size
    assert res.n_trials >= 3  # >= 1 combo evaluated per fold
    assert isinstance(res.oos_equity, pd.Series)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_walkforward.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'fdq.validation.walkforward'`

- [ ] **Step 3: Write minimal implementation**

```python
# fdq/validation/walkforward.py
"""Walk-forward validation: grid-search on train windows, evaluate OOS. No leakage."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from fdq.backtest.engine import BacktestConfig, run_backtest
from fdq.frictions.config import FrictionConfig
from fdq.strategies.benchmarks import build_strategy
from fdq.validation.metrics import sharpe


@dataclass(frozen=True)
class Fold:
    train_start: date
    train_end: date
    test_start: date
    test_end: date


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series
    oos_equity: pd.Series
    fold_params: list[dict[str, Any]]
    trial_sharpes: np.ndarray
    n_trials: int


def make_folds(index: pd.DatetimeIndex, n_folds: int, scheme: str = "expanding") -> list[Fold]:
    idx = index.sort_values()
    n = len(idx)
    if n_folds < 1 or n < n_folds + 1:
        msg = "not enough observations for requested folds"
        raise ValueError(msg)
    block = n // (n_folds + 1)
    folds: list[Fold] = []
    for k in range(1, n_folds + 1):
        train_lo = 0 if scheme == "expanding" else (k - 1) * block
        train_hi = k * block - 1
        test_lo = k * block
        test_hi = min((k + 1) * block - 1, n - 1)
        folds.append(
            Fold(
                idx[train_lo].date(),
                idx[train_hi].date(),
                idx[test_lo].date(),
                idx[test_hi].date(),
            )
        )
    return folds


def grid_combos(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(grid[k] for k in keys))]


def walk_forward(
    strategy_name: str,
    base_params: dict[str, Any],
    grid: dict[str, list[Any]],
    bars: pd.DataFrame,
    tier: float,
    friction: FrictionConfig,
    macro: pd.DataFrame | None,
    n_folds: int = 4,
    scheme: str = "expanding",
    seed: int = 42,
) -> WalkForwardResult:
    folds = make_folds(pd.DatetimeIndex(bars.index), n_folds, scheme)
    combos = grid_combos(grid)
    oos_returns_parts: list[pd.Series] = []
    fold_params: list[dict[str, Any]] = []
    trial_sharpes: list[float] = []

    for fold in folds:
        best_sr = -np.inf
        best_params: dict[str, Any] = combos[0]
        for combo in combos:
            params = {**base_params, **combo}
            strat = build_strategy(strategy_name, params)
            cfg = BacktestConfig(starting_capital=tier, friction=friction, seed=seed)
            res = run_backtest(strat, bars, cfg, macro, fold.train_start, fold.train_end)
            sr = sharpe(res.returns)
            trial_sharpes.append(sr)
            if sr > best_sr:
                best_sr = sr
                best_params = params
        fold_params.append(best_params)
        strat = build_strategy(strategy_name, best_params)
        cfg = BacktestConfig(starting_capital=tier, friction=friction, seed=seed)
        oos = run_backtest(strat, bars, cfg, macro, fold.test_start, fold.test_end)
        oos_returns_parts.append(oos.returns)

    oos_returns = pd.concat(oos_returns_parts).sort_index()
    oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
    oos_equity = tier * (1.0 + oos_returns).cumprod()
    arr = np.array(trial_sharpes, dtype=float)
    return WalkForwardResult(oos_returns, oos_equity, fold_params, arr, arr.size)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_walkforward.py -q && uv run ruff check fdq tests && uv run mypy fdq`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fdq/validation/walkforward.py tests/test_walkforward.py
git commit -m "feat(validation): walk-forward harness with no-leakage folds"
```

---

### Task 1.6: CSCV Probability of Backtest Overfitting

**Files:**
- Modify: `fdq/validation/pbo.py` (replace the stub)
- Test: `tests/test_pbo.py`

**Interfaces:**
- Produces: `probability_backtest_overfitting(returns_matrix: np.ndarray, n_splits: int = 16) -> float` — `returns_matrix` shape `(T, N)` = per-period returns for each of N configurations over the same T observations. Returns PBO ∈ [0, 1].

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pbo.py
from __future__ import annotations

import numpy as np

from fdq.validation.pbo import probability_backtest_overfitting


def test_dominant_config_has_low_pbo() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01, size=(500, 8))
    noise[:, 0] += 0.02  # column 0 dominates in and out of sample
    pbo = probability_backtest_overfitting(noise, n_splits=10)
    assert pbo < 0.3


def test_pure_noise_has_midrange_pbo() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, size=(500, 8))
    pbo = probability_backtest_overfitting(noise, n_splits=10)
    assert 0.3 <= pbo <= 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pbo.py -q`
Expected: FAIL — `NotImplementedError` (current stub).

- [ ] **Step 3: Write minimal implementation**

```python
# fdq/validation/pbo.py
"""Probability of Backtest Overfitting via CSCV (Bailey, Borwein, Lopez de Prado, Zhu)."""

from __future__ import annotations

import itertools

import numpy as np


def _sharpe_cols(block: np.ndarray) -> np.ndarray:
    mean = block.mean(axis=0)
    std = block.std(axis=0, ddof=1)
    std = np.where(std < 1e-12, np.nan, std)
    return mean / std


def probability_backtest_overfitting(returns_matrix: np.ndarray, n_splits: int = 16) -> float:
    m = np.asarray(returns_matrix, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        msg = "returns_matrix must be (T, N) with N >= 2 configurations"
        raise ValueError(msg)
    s = n_splits if n_splits % 2 == 0 else n_splits - 1
    t = m.shape[0]
    if s < 2 or t < s:
        msg = "n_splits must be even, >= 2, and <= number of observations"
        raise ValueError(msg)
    blocks = np.array_split(m[: t - (t % s)], s, axis=0)
    half = s // 2
    logits: list[float] = []
    for combo in itertools.combinations(range(s), half):
        is_idx = list(combo)
        oos_idx = [i for i in range(s) if i not in combo]
        is_block = np.vstack([blocks[i] for i in is_idx])
        oos_block = np.vstack([blocks[i] for i in oos_idx])
        is_perf = _sharpe_cols(is_block)
        oos_perf = _sharpe_cols(oos_block)
        best = int(np.nanargmax(is_perf))
        order = np.argsort(np.argsort(oos_perf))  # ascending ranks
        rank = order[best] / (len(oos_perf) - 1)  # in [0, 1]
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(float(np.log(rank / (1 - rank))))
    if not logits:
        return 0.5
    return float(np.mean(np.array(logits) <= 0.0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pbo.py -q && uv run ruff check fdq tests && uv run mypy fdq`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fdq/validation/pbo.py tests/test_pbo.py
git commit -m "feat(validation): implement CSCV probability of backtest overfitting"
```

---

### Task 1.7: In-sample trial matrix helper for PBO/DSR

**Files:**
- Modify: `fdq/validation/walkforward.py` (add `in_sample_return_matrix`)
- Test: `tests/test_walkforward.py` (add one test)

**Interfaces:**
- Produces: `in_sample_return_matrix(strategy_name, base_params, grid, bars, tier, friction, macro, is_start, is_end, seed=42) -> tuple[np.ndarray, np.ndarray]` — returns `(returns_matrix (T, N), trial_sharpes (N,))` by running every grid combo over the full in-sample window on a common date index (inner-joined, forward-filled to align). Feeds `probability_backtest_overfitting` and DSR.

- [ ] **Step 1: Write the failing test** (append to `tests/test_walkforward.py`)

```python
def test_in_sample_matrix_shape() -> None:
    from fdq.validation.walkforward import in_sample_return_matrix

    spy = pd.read_parquet(FIX)
    bars = pd.concat({"SPY": spy}, axis=1)
    start = bars.index.min().date()
    end = bars.index.max().date()
    mat, sr = in_sample_return_matrix(
        "ma_crossover", {"symbol": "SPY"}, {"fast": [10, 20], "slow": [50]},
        bars, 5.0, FrictionConfig(), None, start, end,
    )
    assert mat.shape[1] == 2 and sr.shape[0] == 2
    assert mat.shape[0] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_walkforward.py::test_in_sample_matrix_shape -q`
Expected: FAIL — `ImportError: cannot import name 'in_sample_return_matrix'`

- [ ] **Step 3: Write minimal implementation** (append to `fdq/validation/walkforward.py`)

```python
def in_sample_return_matrix(
    strategy_name: str,
    base_params: dict[str, Any],
    grid: dict[str, list[Any]],
    bars: pd.DataFrame,
    tier: float,
    friction: FrictionConfig,
    macro: pd.DataFrame | None,
    is_start: date,
    is_end: date,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    combos = grid_combos(grid)
    cols: list[pd.Series] = []
    sharpes: list[float] = []
    for combo in combos:
        params = {**base_params, **combo}
        strat = build_strategy(strategy_name, params)
        cfg = BacktestConfig(starting_capital=tier, friction=friction, seed=seed)
        res = run_backtest(strat, bars, cfg, macro, is_start, is_end)
        cols.append(res.returns.rename(str(combo)))
        sharpes.append(sharpe(res.returns))
    matrix = pd.concat(cols, axis=1).fillna(0.0)
    return matrix.to_numpy(dtype=float), np.array(sharpes, dtype=float)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_walkforward.py -q && uv run ruff check fdq tests && uv run mypy fdq`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fdq/validation/walkforward.py tests/test_walkforward.py
git commit -m "feat(validation): in-sample trial matrix for PBO/DSR"
```

---

### Task 1.8: Walk-forward experiment mode in the runner

**Files:**
- Modify: `fdq/experiment/runner.py` (add `mode: walkforward` branch)
- Create: `experiments/EXP-002/config.yaml` (smoke-scale config used by the test)
- Test: `tests/test_runner_walkforward.py`

**Interfaces:**
- Consumes: `walk_forward`, `in_sample_return_matrix` (Tasks 1.5/1.7); `probability_backtest_overfitting` (1.6); `deflated_sharpe` (existing).
- Produces: experiment configs may declare `mode: walkforward` with, per strategy entry, a `grid` and `n_folds`; the runner computes OOS metrics, real DSR (from `trial_sharpes`), and PBO (from the in-sample matrix), and writes `summary.json` + `report.md`. Backward compatible: absent `mode` (or `mode: fixed`) keeps the existing capital-sweep path.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner_walkforward.py
from __future__ import annotations

import json
from pathlib import Path

import yaml

from fdq.experiment.runner import run_experiment


def test_walkforward_experiment_produces_dsr_and_pbo(tmp_path: Path) -> None:
    fixtures = Path(__file__).parent / "fixtures"
    cfg = {
        "id": "EXP-TEST",
        "title": "walk-forward smoke",
        "friction_version": "1.0.0",
        "mode": "walkforward",
        "n_folds": 3,
        "strategies": [
            {"ma_crossover": {"symbol": "SPY", "grid": {"fast": [10, 20], "slow": [50]}}},
        ],
        "capital_tiers": [5, 50],
        "window": {"start": "2020-01-01", "end": "2020-12-31"},
        "stress_multipliers": [1],
        "seed": 42,
    }
    exp_dir = tmp_path / "EXP-TEST"
    exp_dir.mkdir()
    (exp_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    run_experiment(exp_dir / "config.yaml", tearsheet=False, report=True, data_dir=fixtures)
    summary = json.loads((exp_dir / "results" / "summary.json").read_text())
    tier5 = summary["strategies"][0]["tiers"]["5"]
    assert "dsr" in tier5 and "pbo" in tier5
    assert 0.0 <= tier5["pbo"] <= 1.0
    assert (exp_dir / "report.md").exists()
```

Note: this test requires SPY (and any referenced symbol) fixtures under `tests/fixtures/`. If the fixture window is shorter than the config window, the runner clamps to available dates.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runner_walkforward.py -q`
Expected: FAIL — runner has no `walkforward` branch (`KeyError`/missing `pbo`).

- [ ] **Step 3: Write minimal implementation**

Add near the top of `fdq/experiment/runner.py` imports:

```python
from fdq.validation.pbo import probability_backtest_overfitting
from fdq.validation.walkforward import in_sample_return_matrix, walk_forward
```

Add a dispatch at the start of `run_experiment` after `seed` is parsed:

```python
    if cfg.get("mode") == "walkforward":
        return _run_walkforward_experiment(cfg, exp_dir, results_dir, bars, macro, start, end, tiers, seed)
```

(place it after `bars`/`macro` are loaded). Then add the function:

```python
def _run_walkforward_experiment(
    cfg: dict[str, Any],
    exp_dir: Path,
    results_dir: Path,
    bars: Any,
    macro: Any,
    start: date,
    end: date,
    tiers: list[float],
    seed: int,
) -> Path:
    n_folds = int(cfg.get("n_folds", 4))
    friction = load_friction_config(stress_multiplier=1.0)
    summary: dict[str, Any] = {
        "id": cfg["id"],
        "title": cfg["title"],
        "friction_model_version": cfg.get("friction_version", "1.0.0"),
        "window": cfg["window"],
        "mode": "walkforward",
        "strategies": [],
    }
    report_runs: list[dict[str, Any]] = []
    for entry in cfg["strategies"]:
        name, params = _parse_strategy_entry(entry)
        grid = dict(params.pop("grid", {}))
        tier_stats: dict[str, Any] = {}
        matrix, trial_sharpes = in_sample_return_matrix(
            name, params, grid, bars, tiers[0], friction, macro, start, end, seed
        )
        pbo = probability_backtest_overfitting(matrix) if matrix.shape[1] >= 2 else 0.0
        for tier in tiers:
            wf = walk_forward(name, params, grid, bars, tier, friction, macro, n_folds=n_folds, seed=seed)
            eq = wf.oos_equity
            ret = wf.oos_returns
            tier_stats[str(int(tier))] = {
                "starting_equity": tier,
                "ending_equity": float(eq.iloc[-1]) if len(eq) else tier,
                "cagr": cagr(eq),
                "sharpe": sharpe(ret),
                "sortino": sortino(ret),
                "max_drawdown": max_drawdown(eq),
                "dsr": deflated_sharpe(ret, trial_sharpes),
                "pbo": pbo,
                "n_trials": wf.n_trials,
                "fold_params": wf.fold_params,
            }
            eq.to_frame("equity").to_parquet(
                results_dir / f"{name}_{params.get('symbol', 'combo')}_tier{int(tier)}_equity.parquet"
            )
        label = f"{name}_{params.get('symbol', 'combo')}"
        summary["strategies"].append({"name": label, "params": params, "tiers": tier_stats})
        report_runs.append({"label": f"{name} ({params})", "tiers": tier_stats})

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    _write_walkforward_report(cfg, report_runs, exp_dir / "report.md")
    return results_dir
```

Add the report writer:

```python
def _write_walkforward_report(cfg: dict[str, Any], runs: list[dict[str, Any]], path: Path) -> None:
    lines = [
        f"# {cfg['id']} — {cfg['title']}",
        "",
        "## Methodology",
        "",
        f"- Mode: walk-forward ({cfg.get('n_folds', 4)} expanding folds), grid-search on train only",
        f"- Window (in-sample): {cfg['window']['start']} → {cfg['window']['end']}",
        f"- Friction model version: {cfg.get('friction_version', '1.0.0')}",
        "- Every reported result is net-of-friction; DSR deflates by trial count; PBO via CSCV.",
        "",
        "## Results",
        "",
    ]
    for run in runs:
        lines.append(f"### {run['label']}")
        lines.append("")
        lines.append("| Tier | CAGR | Sharpe | Max DD | DSR | PBO | Trials |")
        lines.append("|------|------|--------|--------|-----|-----|--------|")
        for tier, s in sorted(run["tiers"].items(), key=lambda kv: int(kv[0])):
            lines.append(
                f"| ${tier} | {s['cagr']:.2%} | {s['sharpe']:.2f} | {s['max_drawdown']:.2%} "
                f"| {s['dsr']:.3f} | {s['pbo']:.3f} | {s['n_trials']} |"
            )
        lines.append("")
    lines += ["## Limitations", "", "- Locked test set (2024-2026) not used here.", ""]
    path.write_text("\n".join(lines))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_runner_walkforward.py tests/test_engine.py -q && uv run ruff check fdq tests && uv run mypy fdq`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fdq/experiment/runner.py tests/test_runner_walkforward.py
git commit -m "feat(experiment): walk-forward experiment mode with real DSR + PBO"
```

**CHECKPOINT 1** — Full suite green (`uv run pytest`), ruff + mypy clean, smoke passes (`uv run fdq smoke`). Write a short checkpoint summary (what shipped, any deviations) before Milestone 2.

---

# MILESTONE 2 — Phase 1 experiments (EXP-002, EXP-003)

Expands into full bite-sized tasks at CHECKPOINT 1 (uses Milestone-1 interfaces, now concrete). Task roadmap:

- **Task 2.1 — Refresh CI fixtures for full walk-forward:** ensure `tests/fixtures/` holds enough real history (≥ 400 trading days) for the walk-forward tests; extend `fdq/data/fixtures.py` slice window if needed. Test: fixture provenance + length assertion.
- **Task 2.2 — EXP-002 (trend):** author `experiments/EXP-002/config.yaml` (`mode: walkforward`; `ma_crossover` grids (10,20)/(20,50)/(50,200), `donchian` 20/50; capital tiers; stress 1/2/5). Run via `fdq experiment run`; commit `report.md`, `summary.json`, tearsheet. Deliverable: a real net-of-friction capital-viability table for trend.
- **Task 2.3 — EXP-003 (mean-reversion):** `experiments/EXP-003/config.yaml` with z-score/RSI/Bollinger grids; same protocol. Deliverable: mean-reversion capital-viability report.
- **Task 2.4 — Power-analysis note:** small `validation/power.py` helper (minimum detectable Sharpe given trade count over 6/12/24 months) + a short section appended to each report. Test: known-input MDS value.

Each task: config/data first, then run, then commit artifacts; ruff+mypy+pytest gates apply. **CHECKPOINT 2** before Milestone 3.

---

# MILESTONE 3 — Phase 2 research (HMM regime + cointegration stat-arb)

Adds dependency `statsmodels`. Expands into full tasks at CHECKPOINT 2. Task roadmap:

- **Task 3.1 — In-repo Gaussian HMM (`fdq/regime/hmm.py`):** `GaussianHMM(n_states, seed).fit(X: np.ndarray) -> self`, `.predict_states(X) -> np.ndarray` (Viterbi), `.posteriors(X) -> np.ndarray`; EM/Baum-Welch, numpy/scipy only, deterministic given seed. Test: recovers two well-separated regimes on a controlled numeric sequence (labeled), determinism across two fits with same seed. (No market data in the unit test.)
- **Task 3.2 — Regime features + `RegimeGatedStrategy` (`fdq/strategies/regime_gated.py`):** fit HMM on {log return, realized vol} from real bars up to `asof`; map states→{trend-ok, reversion-ok}; wrap a Tier-1 strategy, gating entries by regime. Test on real fixture: gated strategy never holds when the inner strategy is flat; gate changes at least one entry vs ungated.
- **Task 3.3 — EXP-004 regime ablation:** config running each Tier-1 survivor with and without the gate; report the delta (Sharpe/DSR with vs without). Deliverable: honest "does regime gating help?" verdict.
- **Task 3.4 — Cointegration stat-arb (`fdq/strategies/statarb.py`):** Engle-Granger (`statsmodels.tsa.stattools.coint`/`adfuller`) + Johansen (`statsmodels.tsa.vector_ar.vecm.coint_johansen`) on XOM/CVX, KO/PEP, SPY/IVV; hedge ratio via OLS on train slice; long-only overweight/underweight realization; report the capital tier at which the pair becomes executable under the $1 minimum. Extend `config/universe.yaml` + fetch real bars for the new symbols (provenance sidecars). Test: detects cointegration on a constructed known-cointegrated series; executability-by-tier field present.
- **Task 3.5 — EXP-005 stat-arb frontier:** config + report across capital tiers. Deliverable: stat-arb capital-viability + executability finding.

**CHECKPOINT 3** before Milestone 4.

---

# MILESTONE 4 — Live interactive dashboard (FastAPI + HTMX + Plotly)

Adds dependencies `fastapi`, `uvicorn[standard]`, `python-multipart`, `plotly`. Expands into full tasks at CHECKPOINT 3. Task roadmap:

- **Task 4.1 — Add dashboard deps + `dashboard` optional-extra** in `pyproject.toml`; update `uv.lock` (`uv sync`). Verify import. Update `.github/workflows` if extras needed for route tests.
- **Task 4.2 — Service layer (`fdq/dashboard/service.py`):** in-memory validated-bars cache (loaded once); `run_backtest_spec(strategy, params, tier, stress) -> dict` (serializable: equity, returns, metrics, cost ledger, trades, rejections); `capital_sweep_spec(...) -> dict`; **locked-test guard** — reject any window overlapping 2024-06-01→2026-06-01. Test: spec runs on fixtures; guard raises on locked dates.
- **Task 4.3 — Plotly figure builders (`fdq/dashboard/figures.py`):** functions returning Plotly figure JSON — `capital_viability_fig`, `cost_ledger_fig`, `gross_vs_net_fig`, `equity_bands_fig`, `drawdown_fig`, `regime_timeline_fig`. Shared with the static tearsheet. Test: each returns a dict with `data`/`layout`.
- **Task 4.4 — FastAPI app + templates (`fdq/dashboard/app.py`, `templates/`, `static/`):** routes `GET /`, `GET /experiment/{id}`, `GET /explore`, `POST /api/backtest`, `GET /healthz`; Jinja templates with HTMX (CDN) and Plotly (CDN); responsive CSS grid; restrained institutional styling. Test (FastAPI `TestClient`): every route → 200; `/api/backtest` runs a real fixture-backed backtest and returns metrics; locked-test window → error response.
- **Task 4.5 — `fdq dashboard serve` CLI:** `dashboard` command group + `serve --host --port` launching uvicorn. Test: CLI wiring via `click.testing.CliRunner` (invoke `--help`).
- **Task 4.6 — README + docs:** dashboard section, screenshots note, `updatedplan.md` status update marking Phases 1–2 + dashboard done.

**CHECKPOINT 4 (final)** — full suite green, ruff+mypy clean, `fdq dashboard serve` launches and `/explore` re-runs a backtest live. Then invoke `superpowers:finishing-a-development-branch`.

---

## Self-Review

- **Spec coverage:** trend/mean-rev (M1 T1.2–1.3), walk-forward (T1.5), PBO (T1.6), DSR wiring (T1.7–1.8), HMM regime + gating (M3 T3.1–3.3), stat-arb (T3.4–3.5), EXP-002…005 (T2.2–2.3, T3.3, T3.5), dashboard service/figures/app/CLI (M4), data policy + locked-test guard (Global Constraints + T4.2), reproducibility/seed (BacktestConfig seed threaded through). All spec sections map to a task.
- **Placeholder scan:** Milestone 1 tasks contain complete runnable test + implementation code. Milestones 2–4 are explicit task roadmaps (files, interfaces, algorithms, test intent) that expand into bite-sized steps at their checkpoints — this is a deliberate milestone-checkpoint structure, not an unfilled placeholder, because each later milestone depends on exact signatures produced by the previous one.
- **Type consistency:** `walk_forward`/`in_sample_return_matrix`/`WalkForwardResult.trial_sharpes` (np.ndarray) feed `deflated_sharpe(returns, trial_sharpes: np.ndarray)` and `probability_backtest_overfitting(returns_matrix, n_splits)` consistently; `build_strategy(name, params)` slugs match `spec.slug` values registered in Task 1.4; the long/flat `_LongFlat` contract is shared by trend and mean-reversion.
