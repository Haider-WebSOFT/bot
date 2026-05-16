"""Opportunity ranking and OpportunityCard dataclass."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trading_system.config.constants import Direction, Regime


@dataclass
class OpportunityCard:
    """Ranked trading opportunity ready for execution consideration."""
    symbol: str
    asset_type: str           # "crypto" or "stock"
    direction: Direction      # Always LONG
    strategy_type: str        # "standard" or "early_momentum"
    regime: Regime
    regime_confidence: float
    ensemble_score: float
    agent_scores: dict[str, float]
    suggested_entry: float
    suggested_sl: float
    suggested_tp: float
    r_r_ratio: float
    ev_score: float
    liquidity_score: float
    rvol: float
    mtf_alignment: float
    reasoning_summary: str
    timestamp: datetime


class OpportunityRanker:
    """Ranks opportunities by expected value score."""

    def rank(self, opportunities: list[OpportunityCard]) -> list[OpportunityCard]:
        """Compute EV score and return top 10 sorted by EV."""
        for card in opportunities:
            card.ev_score = (
                (card.ensemble_score / 100)
                * (card.r_r_ratio / 3.0)
                * card.regime_confidence
                * card.liquidity_score
                * (card.mtf_alignment / 100)
            ) * 100

        opportunities.sort(key=lambda c: c.ev_score, reverse=True)
        return opportunities[:10]
