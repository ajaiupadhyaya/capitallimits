"""Strategy signal functions — pure, no I/O."""

from fdq.strategies.base import Strategy, StrategySpec
from fdq.strategies.benchmarks import Balanced6040, BuyAndHold, build_strategy

__all__ = ["Balanced6040", "BuyAndHold", "Strategy", "StrategySpec", "build_strategy"]
