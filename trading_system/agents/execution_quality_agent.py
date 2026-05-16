"""Execution quality agent — tracks slippage and fill rates."""
from __future__ import annotations

from collections import deque

import numpy as np

from trading_system.agents.base_agent import AgentOutput, BaseAgent
from trading_system.config.constants import Direction
from trading_system.data.models import MarketSnapshot


class ExecutionQualityAgent(BaseAgent):
    """Evaluates whether conditions allow quality LONG execution."""

    def __init__(self) -> None:
        self.slippage_history: deque[float] = deque(maxlen=50)
        self.fill_rate_history: deque[float] = deque(maxlen=50)
        self.latency_history: deque[float] = deque(maxlen=20)
        self.expected_slippage = 0.0005

    def record_execution(self, expected_price: float, actual_price: float,
                          filled: bool, latency_ms: float) -> None:
        """Record an execution result for future scoring."""
        actual_slip = abs(actual_price - expected_price) / max(expected_price, 1e-9)
        self.slippage_history.append(actual_slip)
        self.fill_rate_history.append(1.0 if filled else 0.0)
        self.latency_history.append(latency_ms)

    def analyze(self, snap: MarketSnapshot, regime) -> AgentOutput:
        if len(self.slippage_history) < 5:
            return AgentOutput(
                agent_name="ExecutionQualityAgent",
                score=50.0, confidence=0.3, direction=None,
                reasoning={"status": "Insufficient history — neutral"},
                feature_values={},
                timestamp=snap.timestamp,
            )

        score = 50.0
        reasoning: dict[str, str] = {}
        avg_slip = float(np.mean(list(self.slippage_history)))
        fill_rate = float(np.mean(list(self.fill_rate_history)))
        avg_latency = float(np.mean(list(self.latency_history)))
        slip_ratio = avg_slip / max(self.expected_slippage, 1e-9)

        if slip_ratio > 3.0:
            score -= 50
            reasoning["slippage"] = f"Slippage {slip_ratio:.1f}x expected DANGER -50"
        elif slip_ratio > 2.0:
            score -= 25
            reasoning["slippage"] = f"Slippage {slip_ratio:.1f}x expected -25"
        elif slip_ratio <= 1.2:
            score += 25
            reasoning["slippage"] = "Slippage within range +25"

        if fill_rate < 0.70:
            score -= 30
            reasoning["fills"] = f"Fill rate {fill_rate*100:.0f}% poor -30"
        elif fill_rate >= 0.95:
            score += 25
            reasoning["fills"] = f"Fill rate {fill_rate*100:.0f}% excellent +25"

        if avg_latency > 500:
            score -= 25
            reasoning["latency"] = f"Latency={avg_latency:.0f}ms dangerous -25"

        score = max(score, 0)
        direction = Direction.LONG if score >= 40 else None
        confidence = min(len(self.slippage_history) / 50, 1.0)
        return AgentOutput(
            agent_name="ExecutionQualityAgent",
            score=self._clamp(score),
            confidence=confidence,
            direction=direction,
            reasoning=reasoning,
            feature_values={
                "avg_slippage": avg_slip,
                "fill_rate": fill_rate,
                "avg_latency_ms": avg_latency,
                "slip_vs_expected": slip_ratio,
            },
            timestamp=snap.timestamp,
        )
