"""Performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

_ANNUALIZATION = 252


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    years = len(equity) / _ANNUALIZATION
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def sharpe(returns: pd.Series, annualize: bool = True) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    if std < 1e-12:
        return 0.0
    sr = float(returns.mean()) / std
    return sr * np.sqrt(_ANNUALIZATION) if annualize else sr


def sortino(returns: pd.Series, annualize: bool = True) -> float:
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        return 0.0
    ds = float(downside.std(ddof=1))
    if ds < 1e-12:
        return 0.0
    sr = float(returns.mean()) / ds
    return sr * np.sqrt(_ANNUALIZATION) if annualize else sr


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def turnover(trades: pd.DataFrame, equity: pd.Series) -> float:
    if trades.empty or equity.empty:
        return 0.0
    traded = trades["notional"].abs().sum()
    avg_eq = float(equity.mean())
    if avg_eq < 1e-9:
        return 0.0
    years = len(equity) / _ANNUALIZATION
    return float(traded / avg_eq / max(years, 1e-9))
