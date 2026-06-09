"""Friction model configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

FRICTION_MODEL_VERSION = "1.0.0"


@dataclass(frozen=True)
class FrictionConfig:
    """Versioned retail broker friction parameters."""

    version: str = FRICTION_MODEL_VERSION
    min_notional: float = 1.00
    cash_buffer_pct: float = 0.10
    settlement_days: int = 1
    stress_multiplier: float = 1.0
    spread_bps_default: float = 5.0
    spread_bps_by_symbol: dict[str, float] = field(default_factory=dict)
    vix_spread_threshold: float = 25.0
    vix_spread_multiplier: float = 1.5
    sec_fee_rate: float = 0.0000278
    finra_taf_per_share: float = 0.000166
    finra_taf_cap: float = 8.30
    sec_fee_minimum: float = 0.01
    finra_taf_minimum: float = 0.01
    ruin_capital_threshold: float = 1.11

    def spread_bps(self, symbol: str, vix: float | None = None) -> float:
        base = self.spread_bps_by_symbol.get(symbol, self.spread_bps_default)
        if vix is not None and vix > self.vix_spread_threshold:
            base *= self.vix_spread_multiplier
        return base * self.stress_multiplier

    def max_deployable(self, equity: float) -> float:
        """Cash available for new positions after buffer."""
        return equity * (1.0 - self.cash_buffer_pct)

    def max_concurrent_positions(self, equity: float) -> int:
        deployable = self.max_deployable(equity)
        return int(deployable // self.min_notional)


def load_friction_config(
    path: Path | str = "config/friction_v1.yaml",
    stress_multiplier: float = 1.0,
) -> FrictionConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return FrictionConfig(
        version=str(raw.get("version", FRICTION_MODEL_VERSION)),
        min_notional=float(raw["min_notional"]),
        cash_buffer_pct=float(raw["cash_buffer_pct"]),
        settlement_days=int(raw["settlement_days"]),
        stress_multiplier=stress_multiplier,
        spread_bps_default=float(raw.get("spread_bps_default", 5.0)),
        spread_bps_by_symbol={k: float(v) for k, v in raw.get("spread_bps_by_symbol", {}).items()},
        vix_spread_threshold=float(raw.get("vix_spread_threshold", 25.0)),
        vix_spread_multiplier=float(raw.get("vix_spread_multiplier", 1.5)),
        sec_fee_rate=float(raw.get("sec_fee_rate", 0.0000278)),
        finra_taf_per_share=float(raw.get("finra_taf_per_share", 0.000166)),
        finra_taf_cap=float(raw.get("finra_taf_cap", 8.30)),
        sec_fee_minimum=float(raw.get("sec_fee_minimum", 0.01)),
        finra_taf_minimum=float(raw.get("finra_taf_minimum", 0.01)),
        ruin_capital_threshold=float(raw.get("ruin_capital_threshold", 1.11)),
    )
