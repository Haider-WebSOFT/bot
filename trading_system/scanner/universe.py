"""Universe manager for crypto and stock symbols."""
from __future__ import annotations

import time

import pandas as pd

from trading_system.config.constants import CRYPTO_UNIVERSE
from trading_system.core.logger import get_logger

logger = get_logger(__name__)

_STOCK_CACHE: tuple[float, list[str]] | None = None
_STOCK_CACHE_TTL = 4 * 3600  # 4 hours


class UniverseManager:
    """Manages the tradeable symbol universe."""

    CRYPTO_DEFAULT = CRYPTO_UNIVERSE

    async def get_tradeable_stocks(self) -> list[str]:
        """Return top 30 liquid S&P 500 stocks via yfinance."""
        global _STOCK_CACHE
        now = time.time()
        if _STOCK_CACHE and now - _STOCK_CACHE[0] < _STOCK_CACHE_TTL:
            return _STOCK_CACHE[1]

        try:
            sp500 = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )[0]
            symbols = sp500["Symbol"].tolist()
        except Exception as exc:
            logger.warning("S&P 500 list fetch failed", extra={"error": str(exc)})
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                       "META", "TSLA", "BRK-B", "JPM", "JNJ"]

        import yfinance as yf
        filtered: list[tuple[str, float]] = []
        for sym in symbols[:100]:
            try:
                info = yf.Ticker(sym).fast_info
                price = getattr(info, "last_price", 0) or 0
                vol = getattr(info, "three_month_average_volume", 0) or 0
                if price > 10 and vol > 1_000_000:
                    filtered.append((sym, vol))
            except Exception:
                continue

        filtered.sort(key=lambda x: x[1], reverse=True)
        result = [s for s, _ in filtered[:30]]
        _STOCK_CACHE = (now, result)
        logger.info("Stock universe updated", extra={"count": len(result)})
        return result
