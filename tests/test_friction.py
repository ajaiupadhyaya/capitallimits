"""Friction model tests."""

from __future__ import annotations

from datetime import date

import pytest

from fdq.frictions.config import FrictionConfig, load_friction_config
from fdq.frictions.emulator import BrokerEmulator, SettlementLedger, _add_business_days


@pytest.fixture
def friction() -> FrictionConfig:
    return load_friction_config()


@pytest.fixture
def broker(friction: FrictionConfig) -> BrokerEmulator:
    return BrokerEmulator(friction)


def test_min_notional_rejection(broker: BrokerEmulator, friction: FrictionConfig) -> None:
    result = broker.validate_order(0.99, "buy", settled_cash=5.0, current_position=0.0, target_delta=0.99)
    assert not result.accepted
    assert result.reason == "below_min_notional"
    assert broker.rejection_count == 1


def test_fee_floor_dominates_micro_ticket(broker: BrokerEmulator) -> None:
    fill = broker.apply_fill(mid_price=100.0, side="sell", notional=1.0, symbol="SPY")
    assert fill.regulatory_fees >= 0.02
    fee_bps = fill.regulatory_fees / 1.0 * 10_000
    assert fee_bps >= 100


def test_t_plus_one_settlement() -> None:
    monday = date(2024, 6, 3)
    tuesday = _add_business_days(monday, 1)
    assert tuesday.weekday() == 1

    ledger = SettlementLedger(cash_settled=0.0)
    ledger.schedule_sale_proceeds(4.50, monday, settlement_days=1)
    assert ledger.cash_settled == 0.0
    ledger.process_settlements(monday)
    assert ledger.cash_settled == 0.0
    ledger.process_settlements(tuesday)
    assert ledger.cash_settled == pytest.approx(4.50)


def test_spread_double_charge(broker: BrokerEmulator, friction: FrictionConfig) -> None:
    mid = 400.0
    buy = broker.apply_fill(mid, "buy", 4.0, "SPY")
    sell = broker.apply_fill(mid, "sell", 4.0, "SPY")
    round_trip_spread = buy.spread_cost + sell.spread_cost
    half_bps = friction.spread_bps("SPY") / 10_000 / 2
    expected = 4.0 * half_bps * 2
    assert round_trip_spread == pytest.approx(expected, rel=1e-6)


def test_stress_multiplier() -> None:
    base = load_friction_config(stress_multiplier=1.0)
    stressed = load_friction_config(stress_multiplier=2.0)
    b1 = BrokerEmulator(base)
    b2 = BrokerEmulator(stressed)
    f1 = b1.apply_fill(100.0, "buy", 2.0, "SPY")
    f2 = b2.apply_fill(100.0, "buy", 2.0, "SPY")
    assert f2.spread_cost == pytest.approx(f1.spread_cost * 2, rel=1e-6)


def test_max_concurrent_positions_at_5(friction: FrictionConfig) -> None:
    assert friction.max_concurrent_positions(5.0) == 4


def test_insufficient_cash_rejection(broker: BrokerEmulator) -> None:
    result = broker.validate_order(3.0, "buy", settled_cash=2.0, current_position=0.0, target_delta=3.0)
    assert not result.accepted
    assert result.reason == "insufficient_settled_cash"


def test_short_not_allowed(broker: BrokerEmulator) -> None:
    result = broker.validate_order(1.0, "sell", settled_cash=5.0, current_position=0.0, target_delta=-1.0)
    assert not result.accepted
    assert result.reason == "short_not_allowed"


def test_vix_spread_widening(friction: FrictionConfig) -> None:
    normal = friction.spread_bps("SPY", vix=20.0)
    stressed = friction.spread_bps("SPY", vix=30.0)
    assert stressed > normal


def test_friction_version() -> None:
    cfg = load_friction_config()
    assert cfg.version == "1.0.0"
