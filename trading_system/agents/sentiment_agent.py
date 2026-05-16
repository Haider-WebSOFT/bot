"""Sentiment agent — contrarian sentiment using Fear & Greed index."""
from __future__ import annotations

import time
from typing import ClassVar

from trading_system.agents.base_agent import AgentOutput, BaseAgent
from trading_system.config.constants import Direction
from trading_system.data.models import MarketSnapshot
from trading_system.core.logger import get_logger

logger = get_logger(__name__)

_FG_CACHE: tuple[float, int] | None = None
_FG_CACHE_TTL = 3600  # 1 hour


class SentimentAgent(BaseAgent):
    """Contrarian sentiment: extreme fear = potential long opportunity."""

    async def fetch_crypto_fear_greed(self) -> int:
        """Fetch Fear & Greed index (0=extreme fear, 100=extreme greed)."""
        global _FG_CACHE
        now = time.time()
        if _FG_CACHE and now - _FG_CACHE[0] < _FG_CACHE_TTL:
            return _FG_CACHE[1]

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.alternative.me/fng/?limit=1",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    data = await resp.json()
                    value = int(data["data"][0]["value"])
                    _FG_CACHE = (now, value)
                    return value
        except Exception as exc:
            logger.warning("Fear & Greed fetch failed",
                           extra={"error": str(exc)})
            return 50  # neutral on failure

    def analyze(self, snap: MarketSnapshot, regime,
                fear_greed: int = 50, vix: float = 18.0) -> AgentOutput:
        score = 30.0  # start neutral-positive for long-only bias
        reasoning: dict[str, str] = {}

        if fear_greed <= 20:
            score += 40
            reasoning["fg"] = f"Extreme fear={fear_greed} — contrarian long +40"
        elif fear_greed <= 35:
            score += 20
            reasoning["fg"] = f"Fear={fear_greed} — mild bullish +20"
        elif fear_greed >= 80:
            score -= 20
            reasoning["fg"] = f"Extreme greed={fear_greed} — caution -20"
        elif fear_greed >= 65:
            score -= 10
            reasoning["fg"] = f"Greed={fear_greed} — mild caution -10"

        if vix > 30:
            score -= 20
            reasoning["vix"] = f"VIX={vix:.1f} high fear -20"
        elif vix > 25:
            score -= 10
            reasoning["vix"] = f"VIX={vix:.1f} elevated -10"

        score = max(score, 0)
        direction = Direction.LONG if score >= 20 else None
        return AgentOutput(
            agent_name="SentimentAgent",
            score=self._clamp(score),
            confidence=0.5,
            direction=direction,
            reasoning=reasoning,
            feature_values={"fear_greed": float(fear_greed), "vix": vix},
            timestamp=snap.timestamp,
        )
