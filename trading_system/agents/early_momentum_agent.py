"""Strategy C: Early Momentum Agent — catches fast upside moves early."""
from __future__ import annotations

from trading_system.agents.base_agent import AgentOutput, BaseAgent
from trading_system.config.constants import Direction, Regime
from trading_system.data.models import MarketSnapshot


class EarlyMomentumAgent(BaseAgent):
    """
    Strategy C: Catches fast upside moves EARLY.
    Designed for TRENDING_BULL regime only.
    Lower score threshold (55 vs 60) for faster entry.
    """

    SCORE_THRESHOLD = 55  # EARLY_MOMENTUM_SCORE_MIN

    def analyze(self, snap: MarketSnapshot, regime) -> AgentOutput:
        tf15 = snap.tf_15m
        tf4 = snap.tf_4h

        # Guard: only in bull regime
        if regime.regime != Regime.TRENDING_BULL:
            return AgentOutput(
                agent_name="EarlyMomentumAgent", score=0.0,
                confidence=0.0, direction=None,
                reasoning={"regime": f"Not TRENDING_BULL ({regime.regime.value}) — inactive"},
                feature_values={}, timestamp=snap.timestamp,
            )

        # Guard: 4H bull structure
        if not (tf4.ema20 > tf4.ema50):
            return AgentOutput(
                agent_name="EarlyMomentumAgent", score=0.0,
                confidence=0.0, direction=None,
                reasoning={"structure": "4H EMA not bullish — skip"},
                feature_values={}, timestamp=snap.timestamp,
            )

        # Guard: no chaos
        if tf15.atr_pct > tf15.atr_avg20 * 2.0:
            return AgentOutput(
                agent_name="EarlyMomentumAgent", score=0.0,
                confidence=0.0, direction=None,
                reasoning={"chaos": "ATR chaos — skip"},
                feature_values={}, timestamp=snap.timestamp,
            )

        score = 0.0
        reasoning: dict[str, str] = {}

        # Condition 1: RSI Acceleration
        rsi_acceleration = tf15.rsi - tf15.rsi_prev
        cond1 = (rsi_acceleration >= 8 and 45 <= tf15.rsi <= 65 and tf15.rsi_prev < 50)
        if cond1:
            score += 35
            reasoning["rsi_accel"] = (
                f"RSI accelerated +{rsi_acceleration:.1f}pts, crossing 50 +35"
            )
        else:
            reasoning["rsi_accel"] = (
                f"RSI accel={rsi_acceleration:.1f} (need>=8), RSI={tf15.rsi:.1f} — not met"
            )

        # Condition 2: Volume Spike
        cond2 = tf15.rvol >= 2.0 and tf15.obv_slope > 0
        if cond2:
            score += 25
            reasoning["vol_spike"] = (
                f"RVOL={tf15.rvol:.2f} + OBV rising — institutional buy +25"
            )
        else:
            reasoning["vol_spike"] = (
                f"RVOL={tf15.rvol:.2f} or OBV not confirming — not met"
            )

        # Condition 3: MACD Confirmation
        cond3 = tf15.macd_hist_slope > 0 and tf15.macd_histogram > -0.001
        if cond3:
            score += 10
            reasoning["macd_confirm"] = "MACD histogram rising, near/above zero +10"
        else:
            reasoning["macd_confirm"] = "MACD not confirming — not met"

        # Liquidity guard
        if snap.liquidity_score < 0.5:
            score = min(score, 20)
            reasoning["liquidity"] = "Low liquidity — score capped at 20"

        direction = Direction.LONG if score >= self.SCORE_THRESHOLD else None
        if direction:
            reasoning["approval"] = f"Early momentum score={score:.0f} — LONG approved"
        else:
            reasoning["approval"] = (
                f"Score={score:.0f} below threshold ({self.SCORE_THRESHOLD}) — no trade"
            )

        confidence = min(score / 100, 1.0)
        return AgentOutput(
            agent_name="EarlyMomentumAgent",
            score=self._clamp(score),
            confidence=confidence,
            direction=direction,
            reasoning=reasoning,
            feature_values={
                "rsi_acceleration": rsi_acceleration,
                "rvol": tf15.rvol,
                "macd_hist_slope": tf15.macd_hist_slope,
                "rsi_current": tf15.rsi,
                "cond1_met": float(cond1),
                "cond2_met": float(cond2),
                "cond3_met": float(cond3),
            },
            timestamp=snap.timestamp,
        )
