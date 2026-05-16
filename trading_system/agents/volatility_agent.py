"""Volatility agent — scores volatility conditions for long entry quality."""
from __future__ import annotations

from trading_system.agents.base_agent import AgentOutput, BaseAgent
from trading_system.config.constants import Direction, Regime
from trading_system.data.models import MarketSnapshot


class VolatilityAgent(BaseAgent):
    """Scores volatility conditions for LONG entry quality."""

    def analyze(self, snap: MarketSnapshot, regime) -> AgentOutput:
        score = 0.0
        reasoning: dict[str, str] = {}
        tf15 = snap.tf_15m
        tf4 = snap.tf_4h

        # BB Squeeze releasing — best entry condition
        if tf15.bb_squeeze and tf15.bb_width > tf15.bb_width_avg20 * 0.9:
            score += 40
            reasoning["squeeze"] = "BB squeeze releasing — expansion starting +40"
        elif tf15.bb_squeeze:
            score += 20
            reasoning["squeeze"] = "BB squeeze active — potential setup +20"

        # ATR relative to average
        if tf4.atr > tf4.atr_avg20 * 1.2:
            score += 25
            reasoning["atr_expand"] = "ATR expanding above average +25"
        elif tf4.atr < tf4.atr_avg20 * 0.8:
            score += 15
            reasoning["atr_compress"] = "ATR compressed — breakout potential +15"

        # Chaos override — hard penalties for LONG entry
        if tf15.atr_pct > tf15.atr_avg20 * 2.0:
            score = 0
            reasoning["chaos"] = "15m ATR chaos — BLOCK long entry"
        elif tf4.atr_pct > tf4.atr_avg20 * 2.0:
            score = max(score - 40, 0)
            reasoning["chaos"] = "4H ATR elevated -40"

        # In RANGING regime: at lower BB = good long entry
        if regime.regime == Regime.RANGING:
            if snap.current_price <= tf15.bb_lower * 1.002:
                score += 30
                reasoning["bb_bounce"] = "At lower BB — mean reversion long +30"

        direction = Direction.LONG if score >= 20 else None
        confidence = min(score / 100, 1.0)
        return AgentOutput(
            agent_name="VolatilityAgent",
            score=self._clamp(score),
            confidence=confidence,
            direction=direction,
            reasoning=reasoning,
            feature_values={
                "bb_squeeze": float(tf15.bb_squeeze),
                "bb_width": tf15.bb_width,
                "atr_pct": tf15.atr_pct,
                "atr_vs_avg": tf15.atr_pct / max(tf15.atr_avg20, 0.001),
            },
            timestamp=snap.timestamp,
        )
