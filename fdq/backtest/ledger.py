"""Account ledger for dollar-notional positions."""

from __future__ import annotations

from dataclasses import dataclass, field

from fdq.frictions.emulator import SettlementLedger


@dataclass
class CostLedger:
    spread_cents: float = 0.0
    sec_fees_cents: float = 0.0
    finra_taf_cents: float = 0.0
    rejections: int = 0

    def add_fill(self, spread: float, sec: float, taf: float) -> None:
        self.spread_cents += spread * 100
        self.sec_fees_cents += sec * 100
        self.finra_taf_cents += taf * 100

    @property
    def total_cents(self) -> float:
        return self.spread_cents + self.sec_fees_cents + self.finra_taf_cents

    @property
    def total_dollars(self) -> float:
        return self.total_cents / 100.0

    def as_dict(self) -> dict[str, float]:
        return {
            "spread_cents": self.spread_cents,
            "sec_fees_cents": self.sec_fees_cents,
            "finra_taf_cents": self.finra_taf_cents,
            "rejections": float(self.rejections),
            "total_cents": self.total_cents,
        }


@dataclass
class AccountLedger:
    starting_capital: float
    settlement: SettlementLedger
    shares: dict[str, float] = field(default_factory=dict)
    cost: CostLedger = field(default_factory=CostLedger)

    def position_value(self, prices: dict[str, float]) -> float:
        total = 0.0
        for sym, qty in self.shares.items():
            if sym in prices and qty > 0:
                total += qty * prices[sym]
        return total

    def mark_to_market(self, prices: dict[str, float]) -> float:
        cash = self.settlement.cash_settled + sum(p.amount for p in self.settlement.pending)
        return cash + self.position_value(prices)
