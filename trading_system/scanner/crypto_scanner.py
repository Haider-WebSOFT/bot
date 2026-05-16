"""Crypto market scanner — runs 24/7."""
from __future__ import annotations

import asyncio
from datetime import datetime

from trading_system.config.constants import Regime
from trading_system.config.settings import settings
from trading_system.core.logger import get_logger
from trading_system.scanner.ranker import OpportunityCard, OpportunityRanker

logger = get_logger(__name__)


class CryptoScanner:
    """Scans crypto universe every SCAN_INTERVAL_MINUTES."""

    def __init__(self) -> None:
        self._ranker = OpportunityRanker()

    async def scan(
        self,
        data_feed,
        regime_detector,
        ensemble,
        all_agents: dict,
        symbols: list[str] | None = None,
    ) -> list[OpportunityCard]:
        """Run full scan and return ranked opportunities."""
        from trading_system.config.constants import CRYPTO_UNIVERSE
        from trading_system.agents.confirmation_agent import ConfirmationAgent

        symbols = symbols or CRYPTO_UNIVERSE
        confirmation_agent = ConfirmationAgent()
        opportunities: list[OpportunityCard] = []

        snaps = await data_feed.get_all_snapshots(symbols, "crypto")

        for snap in snaps:
            try:
                regime = regime_detector.detect(snap)

                # Skip if hold-cash regime
                if regime.regime in (Regime.TRENDING_BEAR, Regime.HIGH_VOL_CHAOS):
                    continue

                # Liquidity pre-filter
                if snap.liquidity_score < settings.LIQUIDITY_MIN_SCORE:
                    continue
                if snap.spread_pct > 0.002:
                    continue
                if snap.tf_15m.rvol < 1.2:
                    continue

                agent_outputs = {}
                for name, agent in all_agents.items():
                    if name == "SentimentAgent":
                        out = agent.analyze(snap, regime, fear_greed=50)
                    else:
                        out = agent.analyze(snap, regime)
                    agent_outputs[name] = out

                confirmation = confirmation_agent.analyze(snap, regime, agent_outputs)
                result = ensemble.evaluate(agent_outputs, confirmation, regime, snap)

                if not result.trade_approved:
                    continue

                from trading_system.risk.position_sizer import PositionSizer
                sizer = PositionSizer()
                try:
                    sizing = sizer.calculate(snap, regime, account_equity=10.0,
                                              strategy_type=result.strategy_type)
                except Exception:
                    continue

                card = OpportunityCard(
                    symbol=snap.symbol,
                    asset_type="crypto",
                    direction=result.direction,
                    strategy_type=result.strategy_type,
                    regime=regime.regime,
                    regime_confidence=regime.confidence,
                    ensemble_score=result.final_score,
                    agent_scores=result.agent_scores,
                    suggested_entry=snap.current_price,
                    suggested_sl=sizing.stop_loss,
                    suggested_tp=sizing.take_profit,
                    r_r_ratio=sizing.r_r_ratio,
                    ev_score=0.0,
                    liquidity_score=snap.liquidity_score,
                    rvol=snap.tf_15m.rvol,
                    mtf_alignment=snap.mtf_alignment_score,
                    reasoning_summary=result.approval_reason,
                    timestamp=snap.timestamp,
                )
                opportunities.append(card)
            except Exception as exc:
                logger.error("Scan failed for symbol",
                             extra={"symbol": snap.symbol, "error": str(exc)})

        ranked = self._ranker.rank(opportunities)
        logger.info("Crypto scan complete",
                    extra={"scanned": len(snaps), "opportunities": len(ranked)})
        return ranked
