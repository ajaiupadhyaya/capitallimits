"""Pre-registered statistical power: minimum detectable Sharpe ratio.

For a one-sided test that the true Sharpe exceeds zero, the sample period Sharpe
has standard error ~= 1/sqrt(n). The smallest period Sharpe distinguishable from
zero at significance ``alpha`` with power ``1 - beta`` is therefore
``(z_{1-alpha} + z_{power}) / sqrt(n)``; annualizing multiplies by
``sqrt(periods_per_year)``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm  # type: ignore[import-untyped]

_TRADING_DAYS_PER_YEAR = 252


def minimum_detectable_sharpe(
    n_obs: int,
    periods_per_year: int = _TRADING_DAYS_PER_YEAR,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Annualized minimum detectable Sharpe for ``n_obs`` return observations."""
    if n_obs < 2:
        return float("inf")
    z = float(norm.ppf(1.0 - alpha)) + float(norm.ppf(power))
    mds_period = z / np.sqrt(n_obs)
    return float(mds_period * np.sqrt(periods_per_year))


def mds_over_horizon(
    months: float,
    periods_per_year: int = _TRADING_DAYS_PER_YEAR,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Annualized minimum detectable Sharpe over a horizon of ``months`` (daily bars)."""
    n_obs = round(months / 12.0 * periods_per_year)
    return minimum_detectable_sharpe(n_obs, periods_per_year, alpha, power)
