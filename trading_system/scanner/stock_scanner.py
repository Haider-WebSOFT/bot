"""Stock market scanner — runs only during market hours."""
from __future__ import annotations

import asyncio
from datetime import datetime

from trading_system.config.constants import Regime
from trading_system.config.settings import settings
from trading_system.core.logger import get_logger
from trading_system.scanner.ranker import OpportunityCard, OpportunityRanker
from trading_system.scanner.universe import UniverseManager

logger = get_logger(__name__)


class StockScanner:
    """Scans stock universe during market hours using yfinance data."""

    def __init__(self) -> None:
        self._ranker = OpportunityRanker()
        self._universe = UniverseManager()

    async def scan(
        self,
        data_feed,
        alpaca_client,
        regime_detector,
        ensemble,
        all_agents: dict,
    ) -> list[OpportunityCard]:
        """Run stock scan — only during market hours."""
        from trading_system.agents.confirmation_agent import ConfirmationAgent
        import yfinance as yf

        try:
            if not await alpaca_client.is_market_open():
                logger.info("Market closed — stock scan skipped")
                return []
        except Exception:
            return []

        symbols = await self._universe.get_tradeable_stocks()
        confirmation_agent = ConfirmationAgent()
        opportunities: list[OpportunityCard] = []

        snaps = await data_feed.get_all_snapshots(symbols, "stock")

        for snap in snaps:
            try:
                # Earnings filter (skip within 5 days)
                try:
                    ticker = yf.Ticker(snap.symbol)
                    cal = ticker.calendar
                    if cal is not None and "Earnings Date" in cal:
                        earnings = cal["Earnings Date"]
                        if hasattr(earnings, "__iter__"):
                            earnings = list(earnings)[0]
                        if hasattr(earnings, "date"):
                            days_to_earnings = (
                                earnings.date() - datetime.utcnow().date()
                            ).days
                            if abs(days_to_earnings) <= 5:
                                continue
                except Exception:
                    pass

                regime = regime_detector.detect(snap)
                if regime.regime in (Regime.TRENDING_BEAR, Regime.HIGH_VOL_CHAOS):
                    continue
                if snap.liquidity_score < settings.LIQUIDITY_MIN_SCORE:
                    continue
                if snap.current_price < 10:
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
                    asset_type="stock",
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
                logger.error("Stock scan error",
                             extra={"symbol": snap.symbol, "error": str(exc)})

        ranked = self._ranker.rank(opportunities)
        logger.info("Stock scan complete",
                    extra={"scanned": len(snaps), "opportunities": len(ranked)})
        return ranked
