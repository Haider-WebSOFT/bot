"""Trend agent — measures bullish trend direction and structural alignment."""
from __future__ import annotations

from datetime import datetime

from trading_system.agents.base_agent import AgentOutput, BaseAgent
from trading_system.config.constants import Direction
from trading_system.data.models import MarketSnapshot


class TrendAgent(BaseAgent):
    """Measures bullish trend direction and structural alignment."""

    def analyze(self, snap: MarketSnapshot, regime) -> AgentOutput:
        score = 0.0
        reasoning: dict[str, str] = {}
        tf4 = snap.tf_4h
        tf1 = snap.tf_1h

        # EMA bull alignment (most important)
        if tf4.ema20 > tf4.ema50 > tf4.ema200:
            score += 35
            reasoning["ema_alignment"] = "4H full bull EMA stack +35"
        elif tf4.ema20 > tf4.ema50:
            score += 15
            reasoning["ema_alignment"] = "4H partial bull EMA +15"
        else:
            reasoning["ema_alignment"] = "No bull EMA structure +0"

        # ADX trend strength
        if tf4.adx > 30:
            score += 30
            reasoning["adx"] = f"ADX={tf4.adx:.1f} strong trend +30"
        elif tf4.adx > 20:
            score += 15
            reasoning["adx"] = f"ADX={tf4.adx:.1f} moderate trend +15"
        else:
            reasoning["adx"] = f"ADX={tf4.adx:.1f} weak trend +0"

        # Price vs VWAP (above = bullish)
        if snap.tf_1h.price_vs_vwap > 0.002:
            score += 20
            reasoning["vwap"] = "Price above 1H VWAP +20"

        # 1H EMA slope confirmation
        if tf1.ema20_slope > 0:
            score += 15
            reasoning["ema_slope"] = "1H EMA20 rising +15"

        direction = Direction.LONG if score >= 30 else None
        confidence = min(score / 100, 1.0)
        return AgentOutput(
            agent_name="TrendAgent",
            score=self._clamp(score),
            confidence=confidence,
            direction=direction,
            reasoning=reasoning,
            feature_values={
                "ema20": tf4.ema20, "ema50": tf4.ema50,
                "ema200": tf4.ema200, "adx": tf4.adx,
                "vwap_pct": snap.tf_1h.price_vs_vwap,
            },
            timestamp=snap.timestamp,
        )
