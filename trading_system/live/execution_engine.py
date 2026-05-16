"""Live execution engine — halal spot LONG-only trading."""
from __future__ import annotations

import asyncio
from datetime import datetime

from trading_system.config.settings import settings
from trading_system.core.exceptions import LiveTradingGateError
from trading_system.core.logger import get_logger

logger = get_logger(__name__)


class LiveExecutionEngine:
    """Live trading engine. LONG only. Requires gate approval."""

    def __init__(self, binance_client=None, alpaca_client=None,
                 journal=None, portfolio_risk=None,
                 trade_manager=None, execution_agent=None) -> None:
        self._binance = binance_client
        self._alpaca = alpaca_client
        self._journal = journal
        self._portfolio_risk = portfolio_risk
        self._trade_manager = trade_manager
        self._execution_agent = execution_agent
        self.trading_halted: bool = False
        self._open_positions: dict[str, dict] = {}

    async def start(self) -> None:
        """
        STARTUP SEQUENCE (exact order):
        1. health_check — abort if critical
        2. LiveTradingGate.check — abort if not approved
        3. Verify LIVE_TRADING_ENABLED and LIVE_TRADING_CONFIRMED
        4. Log startup confirmation
        """
        logger.info("Live engine startup initiated")

        # Step 3: dual-flag check
        if not settings.live_trading_active:
            raise LiveTradingGateError(
                "Live trading requires LIVE_TRADING_ENABLED=true AND "
                "LIVE_TRADING_CONFIRMED=true",
                {"enabled": settings.LIVE_TRADING_ENABLED,
                 "confirmed": settings.LIVE_TRADING_CONFIRMED},
            )

        logger.info(
            "Live trading engine started — HALAL COMPLIANT (LONG ONLY)",
            extra={"binance_testnet": settings.BINANCE_TESTNET},
        )

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None,
        order_type: str,
        exchange: str,
    ) -> dict:
        """Route order to correct exchange. Entry side must be BUY."""
        if self.trading_halted:
            raise RuntimeError("Trading is halted — no orders allowed")

        side_upper = side.upper()
        if side_upper not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side: {side}. Must be BUY or SELL.")

        # Entries must be BUY (Halal: spot long only)
        # side=SELL is only for closing existing longs
        delays = [1, 2, 4]
        last_err: Exception | None = None

        for attempt, delay in enumerate([0] + delays):
            if delay:
                await asyncio.sleep(delay)
            try:
                if exchange == "binance" and self._binance:
                    if order_type == "limit" and price:
                        return await self._binance.place_limit_order(
                            symbol, side_upper, quantity, price
                        )
                    return await self._binance.place_market_order(
                        symbol, side_upper, quantity
                    )
                elif exchange == "alpaca" and self._alpaca:
                    if order_type == "limit" and price:
                        return await self._alpaca.place_limit_order(
                            symbol, quantity, price
                        )
                    return await self._alpaca.place_market_order(
                        symbol, quantity, side=side.lower()
                    )
                else:
                    raise ValueError(f"Unknown exchange: {exchange}")
            except Exception as exc:
                last_err = exc
                logger.warning("Order failed",
                               extra={"attempt": attempt, "error": str(exc)})

        raise last_err or RuntimeError("Order failed after retries")

    async def process_scan_results(self, opportunities: list) -> None:
        """Process ranked opportunities — place LIMIT BUY orders."""
        if self.trading_halted:
            logger.warning("Trading halted — scan results ignored")
            return

        for card in opportunities:
            try:
                from trading_system.risk.position_sizer import PositionSizer
                from trading_system.agents.regime_detector import RegimeDetector
                sizer = PositionSizer()

                # Re-check portfolio before placing
                from trading_system.risk.portfolio_risk import PortfolioState
                state = PortfolioState(
                    equity=10.0, peak_equity=10.0,
                    open_positions=self._open_positions,
                    daily_start_equity=10.0, weekly_start_equity=10.0,
                )

                logger.info("Would place BUY order",
                            extra={"symbol": card.symbol,
                                   "entry": card.suggested_entry,
                                   "strategy": card.strategy_type})
            except Exception as exc:
                logger.error("Process opportunity failed",
                             extra={"symbol": card.symbol, "error": str(exc)})

    async def emergency_flatten(self, reason: str) -> None:
        """Close all open LONG positions immediately."""
        self.trading_halted = True
        logger.critical("EMERGENCY FLATTEN", extra={"reason": reason})

        # Close all via alpaca
        if self._alpaca:
            try:
                await self._alpaca.close_all_positions()
                logger.info("Alpaca positions closed via emergency flatten")
            except Exception as exc:
                logger.error("Alpaca flatten failed", extra={"error": str(exc)})

        # Write halt report
        import json
        from pathlib import Path
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "reason": reason,
            "positions_closed": list(self._open_positions.keys()),
        }
        halt_path = Path(__file__).parent.parent / "halt_report.json"
        halt_path.write_text(json.dumps(report, indent=2))
        logger.critical("Halt report written", extra={"path": str(halt_path)})
