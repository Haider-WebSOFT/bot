"""Momentum agent — measures bullish momentum quality."""
from __future__ import annotations

from trading_system.agents.base_agent import AgentOutput, BaseAgent
from trading_system.config.constants import Direction
from trading_system.data.models import MarketSnapshot


class MomentumAgent(BaseAgent):
    """Measures bullish momentum quality — direction AND acceleration."""

    def analyze(self, snap: MarketSnapshot, regime) -> AgentOutput:
        score = 0.0
        reasoning: dict[str, str] = {}
        tf1 = snap.tf_1h
        tf15 = snap.tf_15m

        # RSI bull zone
        if 50 < tf1.rsi <= 65:
            score += 35
            reasoning["rsi"] = f"RSI={tf1.rsi:.1f} bull zone, not overbought +35"
        elif 65 < tf1.rsi <= 75:
            score += 20
            reasoning["rsi"] = f"RSI={tf1.rsi:.1f} strong but watch overbought +20"
        elif tf1.rsi > 75:
            score += 5
            reasoning["rsi"] = f"RSI={tf1.rsi:.1f} overbought — caution +5"
        elif tf1.rsi > 45:
            score += 10
            reasoning["rsi"] = f"RSI={tf1.rsi:.1f} neutral +10"
        else:
            reasoning["rsi"] = f"RSI={tf1.rsi:.1f} bearish zone +0"

        # MACD histogram acceleration
        if tf1.macd_hist_slope > 0 and tf1.macd_histogram > 0:
            score += 30
            reasoning["macd"] = "MACD histogram rising positive territory +30"
        elif tf1.macd_hist_slope > 0 and tf1.macd_histogram < 0:
            score += 15
            reasoning["macd"] = "MACD recovering — early bullish signal +15"

        # 15m RSI slope confirmation
        if tf15.rsi_slope > 0:
            score += 20
            reasoning["15m_mom"] = "15m RSI rising — momentum confirmed +20"

        # RSI slope on 1H
        if tf1.rsi_slope > 0:
            score += 15
            reasoning["rsi_slope"] = "1H RSI trending up +15"

        direction = Direction.LONG if score >= 30 else None
        confidence = min(score / 100, 1.0)
        return AgentOutput(
            agent_name="MomentumAgent",
            score=self._clamp(score),
            confidence=confidence,
            direction=direction,
            reasoning=reasoning,
            feature_values={
                "rsi": tf1.rsi, "rsi_slope": tf1.rsi_slope,
                "macd_hist": tf1.macd_histogram,
                "macd_hist_slope": tf1.macd_hist_slope,
            },
            timestamp=snap.timestamp,
        )
