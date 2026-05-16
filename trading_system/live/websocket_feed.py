"""Real-time WebSocket feed manager."""
from __future__ import annotations

import asyncio
from typing import Callable

from trading_system.core.exceptions import HealthCheckError
from trading_system.core.logger import get_logger

logger = get_logger(__name__)


class WebSocketFeedManager:
    """Manages real-time market data streams."""

    def __init__(self) -> None:
        self._consecutive_failures = 0

    async def start_crypto_streams(
        self,
        symbols: list[str],
        on_candle_close: Callable,
    ) -> None:
        """Start Binance WebSocket kline streams."""
        logger.info("Starting crypto streams",
                    extra={"symbols": symbols, "count": len(symbols)})
        # Reconnection logic
        while True:
            try:
                from binance import AsyncClient, BinanceSocketManager
                client = await AsyncClient.create()
                bm = BinanceSocketManager(client)

                streams = [f"{s.lower()}@kline_15m" for s in symbols]
                async with bm.multiplex_socket(streams) as stream:
                    self._consecutive_failures = 0
                    async for msg in stream:
                        try:
                            kline = msg.get("data", {}).get("k", {})
                            if kline.get("x"):  # closed candle
                                await on_candle_close(msg)
                        except Exception as exc:
                            logger.error("Candle handler error",
                                         extra={"error": str(exc)})

            except Exception as exc:
                self._consecutive_failures += 1
                logger.warning("WebSocket disconnected",
                               extra={"failures": self._consecutive_failures,
                                      "error": str(exc)})

                if self._consecutive_failures >= 10:
                    raise HealthCheckError(
                        "10 consecutive WebSocket failures",
                        {"failures": self._consecutive_failures},
                    )

                wait = 30 if self._consecutive_failures >= 3 else 1
                await asyncio.sleep(wait)

    async def start_stock_streams(
        self,
        symbols: list[str],
        on_trade: Callable,
    ) -> None:
        """Start Alpaca trade stream."""
        logger.info("Starting stock streams", extra={"symbols": symbols})
        # Same reconnection pattern as crypto
        self._consecutive_failures = 0
        while True:
            try:
                import alpaca_trade_api as tradeapi
                from trading_system.config.settings import settings
                conn = tradeapi.StreamConn(
                    settings.ALPACA_API_KEY,
                    settings.ALPACA_API_SECRET,
                    base_url=settings.ALPACA_BASE_URL,
                )

                @conn.on(r"^T\..+$")
                async def handle_trade(conn, channel, data):
                    await on_trade(data)

                channels = [f"T.{s}" for s in symbols]
                self._consecutive_failures = 0
                await conn.subscribe(channels)

            except Exception as exc:
                self._consecutive_failures += 1
                logger.warning("Stock stream disconnected",
                               extra={"failures": self._consecutive_failures,
                                      "error": str(exc)})

                if self._consecutive_failures >= 10:
                    raise HealthCheckError(
                        "10 consecutive stock stream failures",
                        {"failures": self._consecutive_failures},
                    )

                wait = 30 if self._consecutive_failures >= 3 else 1
                await asyncio.sleep(wait)
