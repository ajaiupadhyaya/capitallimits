"""Daily backtest engine with dollar-notional positions and T+1 settlement."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import pandas as pd

from fdq.backtest.ledger import AccountLedger, CostLedger
from fdq.frictions.config import FrictionConfig
from fdq.frictions.emulator import BrokerEmulator, SettlementLedger
from fdq.strategies.base import Strategy

ExecutionMode = Literal["next_open"]


@dataclass(frozen=True)
class BacktestConfig:
    starting_capital: float = 5.0
    friction: FrictionConfig | None = None
    execution: ExecutionMode = "next_open"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.friction is None:
            object.__setattr__(self, "friction", FrictionConfig())


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    cost_ledger: CostLedger
    cost_timeseries: pd.DataFrame
    rejections: pd.DataFrame
    config: BacktestConfig
    starting_equity: float
    ending_equity: float
    metadata: dict[str, object] = field(default_factory=dict)


def _to_float(val: object) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _get_price(bars: pd.DataFrame, symbol: str, ts: pd.Timestamp, field: str) -> float | None:
    try:
        col = (symbol, field)
        if col in bars.columns:
            return _to_float(bars.loc[ts, col])
        if symbol in bars.columns.get_level_values(0):
            return _to_float(bars.loc[ts, (symbol, field)])
    except (KeyError, TypeError):
        pass
    return None


def _vix_on(macro: pd.DataFrame | None, ts: pd.Timestamp) -> float | None:
    if macro is None or macro.empty or "vix" not in macro.columns:
        return None
    if ts in macro.index:
        return _to_float(macro.loc[ts, "vix"])
    prior = macro.loc[:ts]
    if prior.empty:
        return None
    return _to_float(prior.iloc[-1]["vix"])


def run_backtest(
    strategy: Strategy,
    bars: pd.DataFrame,
    config: BacktestConfig,
    macro: pd.DataFrame | None = None,
    start: date | None = None,
    end: date | None = None,
) -> BacktestResult:
    friction = config.friction or FrictionConfig()
    broker = BrokerEmulator(friction)
    settlement = SettlementLedger(cash_settled=config.starting_capital)
    ledger = AccountLedger(
        starting_capital=config.starting_capital,
        settlement=settlement,
    )

    dates = bars.index.sort_values()
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    if end:
        dates = dates[dates <= pd.Timestamp(end)]

    equity_records: list[tuple[pd.Timestamp, float]] = []
    cost_records: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    pending_targets: dict[str, float] | None = None

    for i, ts in enumerate(dates):
        asof = ts.date()
        settlement.process_settlements(asof)

        # Execute pending orders from prior signal (next_open)
        if pending_targets is not None and i > 0:
            _execute_rebalance(
                pending_targets,
                ts,
                bars,
                ledger,
                broker,
                friction,
                macro,
                trade_rows,
            )
            pending_targets = None

        # Mark to market at close
        close_prices = {}
        for sym in ledger.shares:
            px = _get_price(bars, sym, ts, "close")
            if px is not None:
                close_prices[sym] = px
        mtm = ledger.mark_to_market(close_prices)
        equity_records.append((ts, mtm))

        cost_records.append(
            {
                "date": ts,
                **ledger.cost.as_dict(),
                "pct_of_starting": ledger.cost.total_cents / (config.starting_capital * 100) * 100,
            }
        )

        # Signal at close, execute next open
        if strategy.should_rebalance(asof, bars):
            weights = strategy.target_weights(asof, bars)
            deployable = friction.max_deployable(mtm)
            pending_targets = {
                sym: float(weights.get(sym, 0.0)) * deployable
                for sym in weights.index
                if float(weights.get(sym, 0.0)) > 0
            }

    equity = pd.Series(
        [e for _, e in equity_records],
        index=pd.DatetimeIndex([t for t, _ in equity_records]),
        name="equity",
    )
    returns = equity.pct_change().fillna(0.0)
    cost_ts = pd.DataFrame(cost_records).set_index("date") if cost_records else pd.DataFrame()
    trades = pd.DataFrame(trade_rows)
    rejections = pd.DataFrame(broker.rejection_log)

    return BacktestResult(
        equity_curve=equity,
        returns=returns,
        trades=trades,
        cost_ledger=ledger.cost,
        cost_timeseries=cost_ts,
        rejections=rejections,
        config=config,
        starting_equity=config.starting_capital,
        ending_equity=float(equity.iloc[-1]) if len(equity) else config.starting_capital,
        metadata={
            "friction_model_version": friction.version,
            "strategy": strategy.spec.slug,
            "execution": config.execution,
        },
    )


def _execute_rebalance(
    targets: dict[str, float],
    ts: pd.Timestamp,
    bars: pd.DataFrame,
    ledger: AccountLedger,
    broker: BrokerEmulator,
    friction: FrictionConfig,
    macro: pd.DataFrame | None,
    trade_rows: list[dict[str, object]],
) -> None:
    vix = _vix_on(macro, ts)
    all_symbols = set(ledger.shares) | set(targets)
    px_open: dict[str, float] = {}
    for sym in all_symbols:
        px = _get_price(bars, sym, ts, "open") or _get_price(bars, sym, ts, "close")
        if px is not None:
            px_open[sym] = px

    current_values = {sym: ledger.shares.get(sym, 0.0) * px_open.get(sym, 0.0) for sym in all_symbols}

    # Sell first to free settled cash
    for sym in sorted(all_symbols):
        current = current_values.get(sym, 0.0)
        target = targets.get(sym, 0.0)
        delta = target - current
        if delta >= -1e-9:
            continue
        sell_notional = min(abs(delta), current)
        if sell_notional < friction.min_notional:
            if sell_notional > 1e-9:
                broker.rejection_count += 1
                ledger.cost.rejections += 1
            continue
        px = px_open.get(sym)
        if px is None:
            continue
        shares_held = ledger.shares.get(sym, 0.0)
        result = broker.validate_order(
            sell_notional, "sell", ledger.settlement.cash_settled, current, delta
        )
        if not result.accepted:
            ledger.cost.rejections += 1
            continue
        fill = broker.apply_fill(px, "sell", sell_notional, sym, vix)
        shares_sold = sell_notional / fill.execution_price
        proceeds = sell_notional - fill.regulatory_fees
        ledger.shares[sym] = shares_held - shares_sold
        if ledger.shares[sym] < 1e-12:
            del ledger.shares[sym]
        ledger.settlement.schedule_sale_proceeds(proceeds, ts.date(), friction.settlement_days)
        ledger.cost.add_fill(fill.spread_cost, fill.sec_fee, fill.finra_taf)
        trade_rows.append(
            {
                "date": ts.date(),
                "symbol": sym,
                "side": "sell",
                "notional": sell_notional,
                "fill_price": fill.execution_price,
                "spread_cost": fill.spread_cost,
                "sec_fee": fill.sec_fee,
                "finra_taf": fill.finra_taf,
            }
        )

    current_values = {sym: ledger.shares.get(sym, 0.0) * px_open.get(sym, 0.0) for sym in all_symbols}
    for sym in sorted(all_symbols):
        current = current_values.get(sym, 0.0)
        target = targets.get(sym, 0.0)
        delta = target - current
        if delta <= 1e-9:
            continue
        buy_notional = delta
        px = px_open.get(sym)
        if px is None:
            continue
        result = broker.validate_order(
            buy_notional, "buy", ledger.settlement.cash_settled, current, delta
        )
        if not result.accepted:
            ledger.cost.rejections += 1
            continue
        fill = broker.apply_fill(px, "buy", buy_notional, sym, vix)
        cash_cost = fill.shares * fill.execution_price
        ledger.settlement.cash_settled -= cash_cost
        ledger.shares[sym] = ledger.shares.get(sym, 0.0) + fill.shares
        ledger.cost.add_fill(fill.spread_cost, fill.sec_fee, fill.finra_taf)
        trade_rows.append(
            {
                "date": ts.date(),
                "symbol": sym,
                "side": "buy",
                "notional": buy_notional,
                "fill_price": fill.execution_price,
                "spread_cost": fill.spread_cost,
                "sec_fee": fill.sec_fee,
                "finra_taf": fill.finra_taf,
            }
        )


def run_capital_sweep(
    make_strategy: Callable[[], Strategy],
    bars: pd.DataFrame,
    tiers: list[float],
    friction: FrictionConfig,
    macro: pd.DataFrame | None = None,
    start: date | None = None,
    end: date | None = None,
    seed: int = 42,
) -> dict[float, BacktestResult]:
    results: dict[float, BacktestResult] = {}
    for tier in tiers:
        cfg = BacktestConfig(
            starting_capital=tier,
            friction=friction,
            seed=seed,
        )
        results[tier] = run_backtest(make_strategy(), bars, cfg, macro, start, end)
    return results
