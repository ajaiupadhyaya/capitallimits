"""Bootstrap analysis for drawdown and ruin probability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fdq.validation.metrics import max_drawdown


@dataclass(frozen=True)
class BootstrapResult:
    ruin_probability: float
    max_drawdown_p5: float
    max_drawdown_p50: float
    max_drawdown_p95: float
    n_samples: int


def bootstrap_ruin_analysis(
    returns: pd.Series,
    starting_capital: float,
    ruin_threshold: float = 1.11,
    n_samples: int = 1000,
    block_size: int = 21,
    seed: int = 42,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    arr = returns.to_numpy(dtype=float)
    n = len(arr)
    if n < block_size:
        return BootstrapResult(0.0, 0.0, 0.0, 0.0, 0)

    max_dds: list[float] = []
    ruins = 0
    for _ in range(n_samples):
        idx: list[int] = []
        while len(idx) < n:
            start = int(rng.integers(0, max(1, n - block_size)))
            idx.extend(range(start, min(start + block_size, n)))
        idx = idx[:n]
        sampled = arr[idx]
        equity = starting_capital * np.cumprod(1.0 + sampled)
        if equity.min() < ruin_threshold:
            ruins += 1
        eq_series = pd.Series(equity)
        max_dds.append(max_drawdown(eq_series))

    dd_arr = np.array(max_dds)
    return BootstrapResult(
        ruin_probability=ruins / n_samples,
        max_drawdown_p5=float(np.percentile(dd_arr, 5)),
        max_drawdown_p50=float(np.percentile(dd_arr, 50)),
        max_drawdown_p95=float(np.percentile(dd_arr, 95)),
        n_samples=n_samples,
    )
