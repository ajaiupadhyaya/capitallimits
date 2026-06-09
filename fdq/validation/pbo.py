"""Probability of Backtest Overfitting (CSCV) — Phase 1.

Full CSCV implementation deferred to Phase 1 when hyperparameter search begins.
"""

from __future__ import annotations

import numpy as np


def probability_backtest_overfitting(
    trial_returns: list[np.ndarray],
) -> float:
    """Stub: CSCV PBO estimator for Phase 1."""
    msg = "PBO/CSCV not implemented until Phase 1 (hyperparameter search). trial_count=1 for Tier 0."
    raise NotImplementedError(msg)
