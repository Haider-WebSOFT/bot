"""Liquidity agent — prevents entering illiquid markets."""
from __future__ import annotations

from trading_system.agents.base_agent import AgentOutput, BaseAgent
from trading_system.config.constants import Direction
from trading_system.data.models import MarketSnapshot


class LiquidityAgent(BaseAgent):
    """Scores market liquidity. Hard blocks on poor liquidity."""

    LIQUIDITY_TIERS: dict[str, float] = {
        "BTCUSDT": 1.0, "ETHUSDT": 0.95,
        "SOLUSDT": 0.85, "BNBUSDT": 0.85, "XRPUSDT": 0.80,
        "ADAUSDT": 0.75, "AVAXUSDT": 0.75, "LINKUSDT": 0.70,
        "DOGEUSDT": 0.70, "MATICUSDT": 0.65,
        "SUIUSDT": 0.60, "LTCUSDT": 0.65,
    }

    def analyze(self, snap: MarketSnapshot, regime) -> AgentOutput:
        score = 0.0
        reasoning: dict[str, str] = {}
        liq = snap.liquidity_score

        # Orderbook depth
        if liq >= 0.80:
            score += 40
            reasoning["orderbook"] = f"Depth={liq:.2f} excellent +40"
        elif liq >= 0.60:
            score += 25
            reasoning["orderbook"] = f"Depth={liq:.2f} adequate +25"
        else:
            reasoning["orderbook"] = f"Depth={liq:.2f} POOR +0"

        # Symbol tier
        symbol_score = self.LIQUIDITY_TIERS.get(snap.symbol, 0.55)
        if symbol_score >= 0.85:
            score += 30
            reasoning["symbol_tier"] = f"{snap.symbol} top-tier +30"
        elif symbol_score >= 0.70:
            score += 20
            reasoning["symbol_tier"] = f"{snap.symbol} mid-tier +20"
        else:
            score += 5
            reasoning["symbol_tier"] = f"{snap.symbol} lower-tier +5"

        # Spread check
        if snap.spread_pct > 0.003:
            score = max(score - 30, 0)
            reasoning["spread"] = f"Spread wide -{30}"
        elif snap.spread_pct < 0.001:
            score += 30
            reasoning["spread"] = f"Spread tight +30"

        # Volume liquidity cross-check
        if snap.tf_15m.rvol < 0.5:
            score = max(score - 30, 0)
            reasoning["vol_liquidity"] = "Very low volume -30"

        # Hard block
        if snap.spread_pct > 0.005 or liq < 0.30:
            score = 0
            reasoning["hard_block"] = "HARD BLOCK: spread or depth below minimum"

        direction = Direction.LONG if score >= 30 else None
        confidence = min(score / 100, 1.0)
        return AgentOutput(
            agent_name="LiquidityAgent",
            score=self._clamp(score),
            confidence=confidence,
            direction=direction,
            reasoning=reasoning,
            feature_values={
                "liquidity_score": liq,
                "spread_pct": snap.spread_pct,
                "symbol_tier": symbol_score,
                "rvol": snap.tf_15m.rvol,
            },
            timestamp=snap.timestamp,
        )
