"""Base agent interface for all signal agents."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from trading_system.config.constants import Direction
from trading_system.data.models import MarketSnapshot


@dataclass
class AgentOutput:
    """Standardised output from any signal agent."""
    agent_name: str
    score: float              # 0 to +100 (LONG only; negative = avoid)
    confidence: float         # 0.0 to 1.0
    direction: Direction | None
    reasoning: dict[str, str]
    feature_values: dict[str, float]
    timestamp: datetime


class BaseAgent(ABC):
    """Abstract base for all trading signal agents."""

    @abstractmethod
    def analyze(self, snap: MarketSnapshot, regime) -> AgentOutput:
        """Analyse the snapshot and return a scoring output."""
        ...

    def _clamp(self, score: float) -> float:
        """LONG-only: floor is 0, ceiling is 100."""
        return max(0.0, min(100.0, score))
