"""Probability of Backtest Overfitting via CSCV (Bailey, Borwein, Lopez de Prado, Zhu)."""

from __future__ import annotations

import itertools

import numpy as np


def _sharpe_cols(block: np.ndarray) -> np.ndarray:
    mean = block.mean(axis=0)
    std = block.std(axis=0, ddof=1)
    std = np.where(std < 1e-12, np.nan, std)
    return np.asarray(mean / std, dtype=float)


def probability_backtest_overfitting(returns_matrix: np.ndarray, n_splits: int = 16) -> float:
    """Estimate PBO from a (T, N) matrix of per-period returns across N configurations.

    Splits the T observations into ``n_splits`` contiguous blocks, forms every
    combinatorial train/test partition of equal halves, and measures how often the
    in-sample-best configuration lands below the out-of-sample median rank.
    """
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
        oos_idx = [i for i in range(s) if i not in combo]
        is_block = np.vstack([blocks[i] for i in combo])
        oos_block = np.vstack([blocks[i] for i in oos_idx])
        is_perf = _sharpe_cols(is_block)
        oos_perf = _sharpe_cols(oos_block)
        best = int(np.nanargmax(is_perf))
        order = np.argsort(np.argsort(oos_perf))  # ascending ranks, 0 = worst
        rank = order[best] / (len(oos_perf) - 1)  # in [0, 1]
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(float(np.log(rank / (1 - rank))))
    if not logits:
        return 0.5
    return float(np.mean(np.array(logits) <= 0.0))
