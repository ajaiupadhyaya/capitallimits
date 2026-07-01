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
    return [
        dict(zip(keys, vals, strict=True))
        for vals in itertools.product(*(grid[k] for k in keys))
    ]


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
