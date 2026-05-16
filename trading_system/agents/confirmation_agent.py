"""Confirmation agent — checks how many core agents agree on LONG."""
from __future__ import annotations

from trading_system.agents.base_agent import AgentOutput, BaseAgent
from trading_system.config.constants import Direction
from trading_system.data.models import MarketSnapshot


class ConfirmationAgent(BaseAgent):
    """Meta-agent: requires minimum core agent agreement for LONG."""

    CORE_AGENTS = ["TrendAgent", "MomentumAgent", "VolumeAgent", "VolatilityAgent"]
    MIN_AGREEING = 3

    def analyze(self, snap: MarketSnapshot, regime,
                agent_outputs: dict[str, AgentOutput] | None = None) -> AgentOutput:
        if agent_outputs is None:
            agent_outputs = {}

        reasoning: dict[str, str] = {}
        directions: dict[str, Direction] = {}

        for name in self.CORE_AGENTS:
            out = agent_outputs.get(name)
            if out and out.direction == Direction.LONG and out.score >= 30:
                directions[name] = Direction.LONG

        agreeing = len(directions)
        total = len(self.CORE_AGENTS)
        agreement_pct = agreeing / total
        score = agreement_pct * 100

        reasoning["agreement"] = f"{agreeing}/{total} core agents confirm LONG"

        if agreeing < self.MIN_AGREEING:
            score = 0
            reasoning["block"] = (
                f"BLOCKED: only {agreeing}/{total} confirm (need {self.MIN_AGREEING})"
            )

        for name in directions:
            reasoning[name] = "LONG confirmed"

        direction = Direction.LONG if agreeing >= self.MIN_AGREEING else None
        return AgentOutput(
            agent_name="ConfirmationAgent",
            score=self._clamp(score),
            confidence=agreement_pct,
            direction=direction,
            reasoning=reasoning,
            feature_values={
                "long_votes": float(agreeing),
                "total_core": float(total),
                "agreement_pct": agreement_pct,
            },
            timestamp=snap.timestamp,
        )
