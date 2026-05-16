"""Self-monitoring loop for the live engine."""
from __future__ import annotations

import asyncio
from datetime import datetime

from trading_system.core.logger import get_logger

logger = get_logger(__name__)


class SelfMonitor:
    """Monitors system health and risk state every 60 seconds."""

    async def run_loop(self, engine, risk_monitor, portfolio_state) -> None:
        """Continuous monitoring loop."""
        while not engine.trading_halted:
            try:
                if risk_monitor.check_emergency_conditions(portfolio_state):
                    await engine.emergency_flatten("Emergency conditions triggered by monitor")
                    break

                logger.info("Monitor tick",
                            extra={"equity": portfolio_state.equity,
                                   "open_positions": len(portfolio_state.open_positions),
                                   "halted": portfolio_state.trading_halted})
            except Exception as exc:
                logger.error("Monitor loop error", extra={"error": str(exc)})

            await asyncio.sleep(60)
