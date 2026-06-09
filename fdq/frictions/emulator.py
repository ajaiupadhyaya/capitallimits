"""Retail broker emulator — versioned friction model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Literal

import pandas as pd

from fdq.frictions.config import FrictionConfig

OrderSide = Literal["buy", "sell"]


class RejectReason(StrEnum):
    BELOW_MIN_NOTIONAL = "below_min_notional"
    INSUFFICIENT_SETTLED_CASH = "insufficient_settled_cash"
    SHORT_NOT_ALLOWED = "short_not_allowed"
    GOOD_FAITH_VIOLATION = "good_faith_violation"


@dataclass(frozen=True)
class OrderResult:
    accepted: bool
    reason: str | None = None


@dataclass(frozen=True)
class Fill:
    execution_price: float
    notional: float
    spread_cost: float
    sec_fee: float
    finra_taf: float
    shares: float

    @property
    def regulatory_fees(self) -> float:
        return self.sec_fee + self.finra_taf

    @property
    def total_friction(self) -> float:
        return self.spread_cost + self.regulatory_fees


@dataclass
class PendingSettlement:
    settle_date: date
    amount: float


@dataclass
class SettlementLedger:
    """Tracks settled vs unsettled cash under T+1 rules."""

    cash_settled: float
    cash_unsettled: float = 0.0
    pending: list[PendingSettlement] = field(default_factory=list)
    good_faith_flags: list[str] = field(default_factory=list)

    def process_settlements(self, asof: date) -> None:
        still_pending: list[PendingSettlement] = []
        for p in self.pending:
            if p.settle_date <= asof:
                self.cash_settled += p.amount
            else:
                still_pending.append(p)
        self.pending = still_pending

    def schedule_sale_proceeds(self, amount: float, trade_date: date, settlement_days: int) -> None:
        settle = _add_business_days(trade_date, settlement_days)
        self.pending.append(PendingSettlement(settle_date=settle, amount=amount))

    @property
    def total_cash(self) -> float:
        pending_sum = sum(p.amount for p in self.pending)
        return self.cash_settled + self.cash_unsettled + pending_sum


def _add_business_days(d: date, n: int) -> date:
    ts = pd.Timestamp(d) + pd.offsets.BDay(n)
    return ts.date()


class BrokerEmulator:
    """Mirrors documented Alpaca retail behavior for simulation."""

    def __init__(self, config: FrictionConfig) -> None:
        self.config = config
        self.rejection_count = 0
        self.rejection_log: list[dict[str, object]] = []
        self.good_faith_flags: list[str] = []

    def validate_order(
        self,
        notional: float,
        side: OrderSide,
        settled_cash: float,
        current_position: float,
        target_delta: float,
    ) -> OrderResult:
        abs_notional = abs(notional)
        if abs_notional < self.config.min_notional and abs_notional > 1e-9:
            return self._reject(RejectReason.BELOW_MIN_NOTIONAL, notional, side)

        if side == "sell" and current_position + target_delta < -1e-9:
            return self._reject(RejectReason.SHORT_NOT_ALLOWED, notional, side)

        if side == "buy" and abs_notional > settled_cash + 1e-9:
            return self._reject(RejectReason.INSUFFICIENT_SETTLED_CASH, notional, side)
            # good-faith: buying with unsettled funds from a same-day sale
        return OrderResult(accepted=True)

    def _reject(self, reason: RejectReason, notional: float, side: OrderSide) -> OrderResult:
        self.rejection_count += 1
        self.rejection_log.append(
            {"reason": reason.value, "notional": notional, "side": side}
        )
        return OrderResult(accepted=False, reason=reason.value)

    def apply_fill(
        self,
        mid_price: float,
        side: OrderSide,
        notional: float,
        symbol: str,
        vix: float | None = None,
    ) -> Fill:
        """Apply half-spread slippage and regulatory fees."""
        if mid_price <= 0:
            msg = "mid_price must be positive"
            raise ValueError(msg)

        spread_bps = self.config.spread_bps(symbol, vix)
        half_spread_frac = (spread_bps / 10_000.0) / 2.0
        if side == "buy":
            execution_price = mid_price * (1.0 + half_spread_frac)
        else:
            execution_price = mid_price * (1.0 - half_spread_frac)

        shares = notional / execution_price
        spread_cost = abs(notional) * half_spread_frac

        sec_fee = 0.0
        finra_taf = 0.0
        if side == "sell":
            sec_fee = max(self.config.sec_fee_minimum, abs(notional) * self.config.sec_fee_rate)
            finra_taf = min(
                self.config.finra_taf_cap,
                max(self.config.finra_taf_minimum, shares * self.config.finra_taf_per_share),
            )

        return Fill(
            execution_price=execution_price,
            notional=notional,
            spread_cost=spread_cost,
            sec_fee=sec_fee,
            finra_taf=finra_taf,
            shares=shares,
        )

    def check_good_faith_violation(
        self, side: OrderSide, had_same_day_sale: bool, using_unsettled: bool
    ) -> bool:
        if side == "buy" and had_same_day_sale and using_unsettled:
            self.good_faith_flags.append("good_faith_violation")
            return True
        return False
