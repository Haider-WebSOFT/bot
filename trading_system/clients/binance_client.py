"""Async Binance client wrapping python-binance."""
import asyncio
from datetime import datetime, timezone

import pandas as pd

from trading_system.config.settings import settings
from trading_system.core.exceptions import (
    AuthenticationError, InsufficientFundsError, OrderError, RateLimitError,
)
from trading_system.core.logger import get_logger

logger = get_logger(__name__)


class BinanceClient:
    """Async wrapper around python-binance Client."""

    def __init__(self) -> None:
        from binance.client import Client

        api_key = settings.BINANCE_API_KEY
        api_secret = settings.BINANCE_API_SECRET
        testnet = settings.BINANCE_TESTNET
        self._client = Client(api_key, api_secret, testnet=testnet)
        logger.info("BinanceClient initialised", extra={"testnet": testnet})

    async def _retry(self, fn, *args, **kwargs):
        delays = [0.5, 1]
        last_err = None
        for attempt, delay in enumerate([0] + delays):
            if delay:
                await asyncio.sleep(delay)
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_err = exc
                self._handle_binance_error(exc, attempt)
        raise last_err

    def _handle_binance_error(self, exc: Exception, attempt: int) -> None:
        try:
            from binance.exceptions import BinanceAPIException
            if isinstance(exc, BinanceAPIException):
                if exc.code == -1003:
                    raise RateLimitError("Binance rate limit", {"code": exc.code})
                if exc.code == -2010:
                    raise InsufficientFundsError("Insufficient funds", {"code": exc.code})
                if exc.code == -2014 or exc.code == -1100:
                    raise AuthenticationError("Binance auth error", {"code": exc.code})
        except ImportError:
            pass
        logger.warning("Binance call failed", extra={"attempt": attempt, "error": str(exc)})

    async def get_ohlcv(self, symbol: str, interval: str,
                         limit: int = 500) -> pd.DataFrame:
        """Fetch OHLCV klines and return as DataFrame."""
        raw = await self._retry(
            self._client.get_klines,
            symbol=symbol, interval=interval, limit=limit,
        )
        df = pd.DataFrame(raw, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        return df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    async def get_orderbook(self, symbol: str, depth: int = 20) -> dict:
        """Fetch order book depth."""
        book = await self._retry(self._client.get_order_book, symbol=symbol, limit=depth)
        bids = [[float(p), float(q)] for p, q in book["bids"]]
        asks = [[float(p), float(q)] for p, q in book["asks"]]
        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 0.0
        spread_pct = (best_ask - best_bid) / best_bid if best_bid else 0.0
        return {"bids": bids, "asks": asks, "spread_pct": spread_pct}

    async def get_account_balance(self) -> dict[str, float]:
        """Return non-zero asset balances."""
        account = await self._retry(self._client.get_account)
        return {
            b["asset"]: float(b["free"])
            for b in account["balances"]
            if float(b["free"]) > 0
        }

    async def place_limit_order(self, symbol: str, side: str,
                                 quantity: float, price: float) -> dict:
        """Place a limit order. Entry side must be BUY."""
        if side.upper() not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side: {side}. Must be BUY or SELL.")
        result = await self._retry(
            self._client.order_limit,
            symbol=symbol,
            side=side.upper(),
            quantity=quantity,
            price=str(price),
        )
        return result

    async def place_market_order(self, symbol: str, side: str,
                                  quantity: float) -> dict:
        """Place a market order."""
        if side.upper() not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side: {side}")
        result = await self._retry(
            self._client.order_market,
            symbol=symbol,
            side=side.upper(),
            quantity=quantity,
        )
        return result

    async def cancel_order(self, symbol: str, order_id: int) -> dict:
        """Cancel an open order."""
        return await self._retry(
            self._client.cancel_order, symbol=symbol, orderId=order_id,
        )

    async def get_order_status(self, symbol: str, order_id: int) -> dict:
        """Get status of an order."""
        return await self._retry(
            self._client.get_order, symbol=symbol, orderId=order_id,
        )
