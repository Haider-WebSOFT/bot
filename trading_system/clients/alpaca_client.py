"""Alpaca paper-trading client with yfinance for OHLCV."""
import asyncio
import time
from datetime import datetime
from functools import lru_cache
from typing import Any

import pandas as pd

from trading_system.config.settings import settings
from trading_system.core.logger import get_logger

logger = get_logger(__name__)

_OHLCV_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_CLOCK_CACHE: tuple[float, Any] | None = None
_YFINANCE_TF_MAP = {
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
}


class AlpacaClient:
    """Alpaca REST client for paper trading; uses yfinance for market data."""

    def __init__(self) -> None:
        import alpaca_trade_api as tradeapi

        self.api = tradeapi.REST(
            settings.ALPACA_API_KEY,
            settings.ALPACA_API_SECRET,
            settings.ALPACA_BASE_URL,
        )
        logger.info("AlpacaClient initialised",
                    extra={"base_url": settings.ALPACA_BASE_URL})

    async def get_ohlcv(self, symbol: str, timeframe: str,
                         limit: int = 500) -> pd.DataFrame:
        """Fetch OHLCV via yfinance (Alpaca free tier has no SIP data)."""
        cache_key = f"{symbol}:{timeframe}"
        now = time.time()
        if cache_key in _OHLCV_CACHE:
            cached_at, df = _OHLCV_CACHE[cache_key]
            if now - cached_at < 900:  # 15-minute cache
                return df.tail(limit).copy()

        import yfinance as yf

        yf_tf = _YFINANCE_TF_MAP.get(timeframe, timeframe)
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="60d", interval=yf_tf)
        if df.empty:
            logger.warning("yfinance returned empty dataframe",
                           extra={"symbol": symbol, "timeframe": timeframe})
            return pd.DataFrame(columns=["timestamp", "open", "high", "low",
                                          "close", "volume"])

        df = df.reset_index()
        ts_col = "Datetime" if "Datetime" in df.columns else "Date"
        df = df.rename(columns={
            ts_col: "timestamp",
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        _OHLCV_CACHE[cache_key] = (now, df)
        return df.tail(limit).copy()

    async def get_account(self) -> dict:
        """Return account summary."""
        account = self.api.get_account()
        return {
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "cash": float(account.cash),
            "equity": float(account.equity),
        }

    async def is_market_open(self) -> bool:
        """Check if market is currently open (cached 60s)."""
        global _CLOCK_CACHE
        now = time.time()
        if _CLOCK_CACHE and now - _CLOCK_CACHE[0] < 60:
            return _CLOCK_CACHE[1]
        clock = self.api.get_clock()
        result = clock.is_open
        _CLOCK_CACHE = (now, result)
        return result

    async def get_market_hours(self) -> dict:
        """Return market hours information."""
        clock = self.api.get_clock()
        return {
            "is_open": clock.is_open,
            "open_time": str(clock.next_open),
            "close_time": str(clock.next_close),
            "next_open": str(clock.next_open),
            "next_close": str(clock.next_close),
        }

    async def place_limit_order(self, symbol: str, qty: float,
                                 limit_price: float) -> dict:
        """Place a limit BUY order (Halal: long only)."""
        order = self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="limit",
            time_in_force="day",
            limit_price=limit_price,
        )
        return {"id": order.id, "status": order.status, "symbol": symbol,
                "qty": qty, "limit_price": limit_price}

    async def place_market_order(self, symbol: str, qty: float,
                                  side: str = "buy") -> dict:
        """Place a market order. Side: 'buy' for entries, 'sell' for closes."""
        order = self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day",
        )
        return {"id": order.id, "status": order.status}

    async def cancel_order(self, order_id: str) -> dict:
        """Cancel an order by ID."""
        self.api.cancel_order(order_id)
        return {"cancelled": order_id}

    async def get_positions(self) -> list[dict]:
        """Return all open positions."""
        positions = self.api.list_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
            }
            for p in positions
        ]

    async def close_all_positions(self) -> None:
        """Emergency: close all open positions."""
        self.api.close_all_positions()
        logger.warning("All positions closed via emergency flatten")
