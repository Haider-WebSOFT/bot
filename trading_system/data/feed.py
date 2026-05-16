"""Data feed: fetches multi-timeframe snapshots from exchanges."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime

import pandas as pd

from trading_system.core.exceptions import InsufficientDataError
from trading_system.core.logger import get_logger
from trading_system.data.indicators import compute_indicator_set
from trading_system.data.models import IndicatorSet, MarketSnapshot
from trading_system.data.multi_timeframe import compute_mtf_alignment

logger = get_logger(__name__)

_SNAPSHOT_CACHE: dict[str, tuple[float, MarketSnapshot]] = {}
_CACHE_TTL = 60  # seconds


class DataFeed:
    """Fetches and assembles multi-timeframe market snapshots."""

    def __init__(self, binance_client, alpaca_client) -> None:
        self._binance = binance_client
        self._alpaca = alpaca_client

    async def get_snapshot(self, symbol: str, asset_type: str) -> MarketSnapshot:
        """Fetch all timeframes, compute indicators, return MarketSnapshot."""
        cache_key = f"{asset_type}:{symbol}"
        now = time.time()
        if cache_key in _SNAPSHOT_CACHE:
            cached_at, snap = _SNAPSHOT_CACHE[cache_key]
            if now - cached_at < _CACHE_TTL:
                return snap

        client = self._binance if asset_type == "crypto" else self._alpaca

        if asset_type == "crypto":
            df_15m, df_1h, df_4h, ob = await asyncio.gather(
                client.get_ohlcv(symbol, "15m", limit=500),
                client.get_ohlcv(symbol, "1h", limit=500),
                client.get_ohlcv(symbol, "4h", limit=500),
                client.get_orderbook(symbol, depth=20),
                return_exceptions=False,
            )
            spread_pct = float(ob.get("spread_pct", 0.001))
            bids = ob.get("bids", [])
            depth_score = sum(b[1] for b in bids[:10]) if bids else 0
            liquidity_score = min(depth_score / 100_000, 1.0)
        else:
            df_15m, df_1h, df_4h = await asyncio.gather(
                client.get_ohlcv(symbol, "15m", limit=500),
                client.get_ohlcv(symbol, "1h", limit=500),
                client.get_ohlcv(symbol, "4h", limit=500),
                return_exceptions=False,
            )
            spread_pct = 0.001
            vol = float(df_15m["volume"].mean()) if not df_15m.empty else 0
            liquidity_score = min(vol / 1_000_000, 1.0)

        ts = datetime.utcnow()

        try:
            ind_15m = compute_indicator_set(df_15m, ts)
        except InsufficientDataError:
            ind_15m = _empty_indicator_set(df_15m, ts)

        try:
            ind_1h = compute_indicator_set(df_1h, ts)
        except InsufficientDataError:
            ind_1h = _empty_indicator_set(df_1h, ts)

        try:
            ind_4h = compute_indicator_set(df_4h, ts)
        except InsufficientDataError:
            ind_4h = _empty_indicator_set(df_4h, ts)

        current_price = float(df_15m["close"].iloc[-1]) if not df_15m.empty else 0.0

        snap = MarketSnapshot(
            symbol=symbol,
            current_price=current_price,
            tf_15m=ind_15m,
            tf_1h=ind_1h,
            tf_4h=ind_4h,
            mtf_alignment_score=0.0,
            spread_pct=spread_pct,
            liquidity_score=liquidity_score,
            timestamp=ts,
        )
        # Compute MTF alignment in-place
        from dataclasses import replace as dc_replace
        snap = MarketSnapshot(
            symbol=snap.symbol,
            current_price=snap.current_price,
            tf_15m=snap.tf_15m,
            tf_1h=snap.tf_1h,
            tf_4h=snap.tf_4h,
            mtf_alignment_score=compute_mtf_alignment(snap),
            spread_pct=snap.spread_pct,
            liquidity_score=snap.liquidity_score,
            timestamp=snap.timestamp,
        )
        _SNAPSHOT_CACHE[cache_key] = (now, snap)
        logger.info("Snapshot fetched", extra={"symbol": symbol, "asset_type": asset_type,
                                                "price": current_price})
        return snap

    async def get_all_snapshots(self, symbols: list[str],
                                 asset_type: str) -> list[MarketSnapshot]:
        """Fetch all symbols concurrently with per-symbol timeout."""
        async def _fetch(sym):
            return await asyncio.wait_for(self.get_snapshot(sym, asset_type), timeout=15.0)

        tasks = [_fetch(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        snaps = []
        for sym, res in zip(symbols, results):
            if isinstance(res, Exception):
                logger.warning("Snapshot skipped", extra={"symbol": sym, "error": str(res)})
            else:
                snaps.append(res)
        return snaps


def _empty_indicator_set(df: pd.DataFrame, ts: datetime) -> IndicatorSet:
    """Returns a zeroed IndicatorSet when data is insufficient."""
    price = float(df["close"].iloc[-1]) if len(df) > 0 else 1.0
    return IndicatorSet(
        ema20=price, ema50=price, ema200=price,
        ema20_slope=0.0, ema50_slope=0.0, price_vs_vwap=0.0,
        rsi=50.0, rsi_prev=50.0, rsi_slope=0.0,
        macd_line=0.0, macd_signal=0.0, macd_histogram=0.0, macd_hist_slope=0.0,
        adx=0.0, di_plus=0.0, di_minus=0.0,
        atr=price * 0.01, atr_pct=1.0, atr_avg20=price * 0.01,
        bb_upper=price * 1.02, bb_lower=price * 0.98, bb_mid=price, bb_width=0.04,
        bb_width_avg20=0.04, bb_squeeze=False,
        obv=0.0, obv_slope=0.0, rvol=1.0, vwap=price,
        resistance_levels=[], support_levels=[],
        nearest_resistance=price * 1.05, nearest_support=price * 0.95,
        timestamp=ts,
    )
