"""Ensemble engine — combines all agent scores into a final LONG decision."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trading_system.agents.base_agent import AgentOutput
from trading_system.agents.regime_detector import RegimeResult
from trading_system.config.constants import Direction, Regime
from trading_system.data.models import MarketSnapshot
from trading_system.data.multi_timeframe import is_counter_trend

AGENT_WEIGHTS_BY_REGIME: dict[Regime, dict[str, float]] = {
    Regime.TRENDING_BULL: {
        "TrendAgent": 0.24, "MomentumAgent": 0.20, "VolumeAgent": 0.16,
        "VolatilityAgent": 0.10, "LiquidityAgent": 0.10,
        "EarlyMomentumAgent": 0.10,
        "SentimentAgent": 0.05, "ExecutionQualityAgent": 0.05,
    },
    Regime.TRENDING_BEAR: {
        "TrendAgent": 0.125, "MomentumAgent": 0.125, "VolumeAgent": 0.125,
        "VolatilityAgent": 0.125, "LiquidityAgent": 0.125,
        "EarlyMomentumAgent": 0.125,
        "SentimentAgent": 0.125, "ExecutionQualityAgent": 0.125,
    },
    Regime.RANGING: {
        "TrendAgent": 0.10, "MomentumAgent": 0.15, "VolumeAgent": 0.15,
        "VolatilityAgent": 0.30, "LiquidityAgent": 0.15,
        "EarlyMomentumAgent": 0.00,
        "SentimentAgent": 0.10, "ExecutionQualityAgent": 0.05,
    },
    Regime.VOL_COMPRESSION: {
        "TrendAgent": 0.15, "MomentumAgent": 0.15, "VolumeAgent": 0.20,
        "VolatilityAgent": 0.30, "LiquidityAgent": 0.10,
        "EarlyMomentumAgent": 0.00,
        "SentimentAgent": 0.05, "ExecutionQualityAgent": 0.05,
    },
    Regime.HIGH_VOL_CHAOS: {
        "TrendAgent": 0.125, "MomentumAgent": 0.125, "VolumeAgent": 0.125,
        "VolatilityAgent": 0.125, "LiquidityAgent": 0.125,
        "EarlyMomentumAgent": 0.125,
        "SentimentAgent": 0.125, "ExecutionQualityAgent": 0.125,
    },
}


@dataclass
class EnsembleResult:
    """Final combined decision from all agents."""
    final_score: float
    final_confidence: float
    direction: Direction | None
    trade_approved: bool
    approval_reason: str
    strategy_type: str          # "standard" | "early_momentum"
    agent_scores: dict[str, float]
    agent_confidences: dict[str, float]
    weights_used: dict[str, float]
    confirmation_result: AgentOutput
    timestamp: datetime


class EnsembleEngine:
    """Combines all agent outputs into a single LONG/None decision."""

    def evaluate(
        self,
        agent_outputs: dict[str, AgentOutput],
        confirmation: AgentOutput,
        regime: RegimeResult,
        snap: MarketSnapshot,
    ) -> EnsembleResult:
        ts = snap.timestamp
        agent_scores = {k: v.score for k, v in agent_outputs.items()}
        agent_confidences = {k: v.confidence for k, v in agent_outputs.items()}
        weights = AGENT_WEIGHTS_BY_REGIME.get(regime.regime,
                                               AGENT_WEIGHTS_BY_REGIME[Regime.RANGING])

        def _block(reason: str, strategy: str = "standard") -> EnsembleResult:
            return EnsembleResult(
                final_score=0.0, final_confidence=0.0,
                direction=None, trade_approved=False,
                approval_reason=reason, strategy_type=strategy,
                agent_scores=agent_scores, agent_confidences=agent_confidences,
                weights_used=weights, confirmation_result=confirmation,
                timestamp=ts,
            )

        # Step 1: TRENDING_BEAR — hold cash
        if regime.regime == Regime.TRENDING_BEAR:
            return _block("TRENDING_BEAR — hold cash, no new longs (Halal)")

        # Step 2: HIGH_VOL_CHAOS — no new positions
        if regime.regime == Regime.HIGH_VOL_CHAOS:
            return _block("HIGH_VOL_CHAOS — no new positions, preserve capital")

        # Step 3: LiquidityAgent hard block
        liq_out = agent_outputs.get("LiquidityAgent")
        if liq_out and liq_out.score == 0 and "hard_block" in liq_out.reasoning:
            return _block("LiquidityAgent hard block")

        # Step 4: ConfirmationAgent block
        if confirmation.direction is None:
            return _block(f"ConfirmationAgent blocked: {confirmation.reasoning.get('block','')}")

        # Step 5: Weighted ensemble scoring
        weighted_score = 0.0
        weighted_conf = 0.0
        total_weight = 0.0

        for name, out in agent_outputs.items():
            w = weights.get(name, 0.0)
            if w == 0.0:
                continue
            weighted_score += out.score * w * out.confidence
            weighted_conf += out.confidence * w
            total_weight += w

        if total_weight > 0:
            weighted_score /= total_weight
            weighted_conf /= total_weight

        # Step 6: Apply regime risk multiplier
        weighted_conf *= regime.risk_multiplier

        # Step 7: Counter-trend check
        if is_counter_trend(snap):
            return _block("Counter-trend: 4H bearish structure blocks LONG")

        # Step 8: Check early momentum path
        em_out = agent_outputs.get("EarlyMomentumAgent")
        early_approved = (
            em_out is not None
            and em_out.direction == Direction.LONG
            and em_out.score >= 55
        )

        # Standard path
        if weighted_score >= 60 and weighted_conf >= 0.65:
            return EnsembleResult(
                final_score=weighted_score,
                final_confidence=weighted_conf,
                direction=Direction.LONG,
                trade_approved=True,
                approval_reason="Standard path approved",
                strategy_type="standard",
                agent_scores=agent_scores,
                agent_confidences=agent_confidences,
                weights_used=weights,
                confirmation_result=confirmation,
                timestamp=ts,
            )

        # Early momentum path
        if early_approved and weighted_score >= 55 and weighted_conf >= 0.55:
            return EnsembleResult(
                final_score=weighted_score,
                final_confidence=weighted_conf,
                direction=Direction.LONG,
                trade_approved=True,
                approval_reason="Early momentum path approved (Strategy C)",
                strategy_type="early_momentum",
                agent_scores=agent_scores,
                agent_confidences=agent_confidences,
                weights_used=weights,
                confirmation_result=confirmation,
                timestamp=ts,
            )

        return _block(
            f"Score {weighted_score:.1f} or confidence {weighted_conf:.2f} below threshold"
        )
