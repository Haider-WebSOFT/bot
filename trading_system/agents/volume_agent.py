"""Volume agent — detects institutional participation."""
from __future__ import annotations

from trading_system.agents.base_agent import AgentOutput, BaseAgent
from trading_system.config.constants import Direction
from trading_system.data.models import MarketSnapshot


class VolumeAgent(BaseAgent):
    """Detects institutional participation via volume analysis."""

    def analyze(self, snap: MarketSnapshot, regime) -> AgentOutput:
        score = 0.0
        reasoning: dict[str, str] = {}
        tf1 = snap.tf_1h
        tf15 = snap.tf_15m

        # Relative Volume — primary signal
        if tf15.rvol >= 2.5:
            score += 50
            reasoning["rvol"] = f"RVOL={tf15.rvol:.2f} extreme institutional volume +50"
        elif tf15.rvol >= 2.0:
            score += 40
            reasoning["rvol"] = f"RVOL={tf15.rvol:.2f} strong +40"
        elif tf15.rvol >= 1.5:
            score += 25
            reasoning["rvol"] = f"RVOL={tf15.rvol:.2f} above average +25"
        elif tf15.rvol < 0.8:
            reasoning["rvol"] = f"RVOL={tf15.rvol:.2f} weak — avoid breakouts +0"

        # OBV slope — accumulation
        if tf1.obv_slope > 0:
            score += 25
            reasoning["obv"] = "OBV rising — accumulation +25"

        # Volume confirms bull trend
        trend_bull = snap.tf_4h.ema20 > snap.tf_4h.ema50
        if tf15.rvol > 1.5 and trend_bull:
            score += 20
            reasoning["vol_confirm"] = "Volume confirms bull trend +20"

        # Breakout gate: low volume caps score
        if tf15.rvol < 1.2:
            score = min(score, 20)
            reasoning["breakout_gate"] = "Low RVOL — breakout not confirmed, score capped at 20"

        direction = Direction.LONG if score >= 30 else None
        confidence = min(score / 100, 1.0)
        return AgentOutput(
            agent_name="VolumeAgent",
            score=self._clamp(score),
            confidence=confidence,
            direction=direction,
            reasoning=reasoning,
            feature_values={
                "rvol_15m": tf15.rvol,
                "obv_slope": tf1.obv_slope,
                "rvol_1h": tf1.rvol,
            },
            timestamp=snap.timestamp,
        )
