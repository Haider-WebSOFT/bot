"""Module 4 data feed tests."""
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_ohlcv(n: int = 300, base: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = base + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "open": close - 0.1,
        "high": close + 0.3,
        "low": close - 0.3,
        "close": close,
        "volume": rng.uniform(1000, 5000, n),
    })


def _make_bull_indicator_set(price: float = 100.0):
    from trading_system.data.models import IndicatorSet

    return IndicatorSet(
        ema20=price, ema50=price * 0.95, ema200=price * 0.85,
        ema20_slope=0.01, ema50_slope=0.005, price_vs_vwap=0.001,
        rsi=58.0, rsi_prev=52.0, rsi_slope=0.02,
        macd_line=0.5, macd_signal=0.3, macd_histogram=0.2, macd_hist_slope=0.05,
        adx=28.0, di_plus=25.0, di_minus=15.0,
        atr=1.0, atr_pct=1.0, atr_avg20=1.0,
        bb_upper=102.0, bb_lower=98.0, bb_mid=100.0, bb_width=0.04,
        bb_width_avg20=0.05, bb_squeeze=False,
        obv=10000.0, obv_slope=0.01, rvol=2.0, vwap=99.5,
        resistance_levels=[105.0], support_levels=[95.0],
        nearest_resistance=105.0, nearest_support=95.0,
        timestamp=datetime.utcnow(),
    )


def _make_bear_indicator_set(price: float = 100.0):
    from trading_system.data.models import IndicatorSet

    return IndicatorSet(
        ema20=price * 0.85, ema50=price * 0.95, ema200=price,
        ema20_slope=-0.01, ema50_slope=-0.005, price_vs_vwap=-0.01,
        rsi=42.0, rsi_prev=48.0, rsi_slope=-0.02,
        macd_line=-0.5, macd_signal=-0.3, macd_histogram=-0.2, macd_hist_slope=-0.05,
        adx=28.0, di_plus=15.0, di_minus=25.0,
        atr=1.0, atr_pct=1.0, atr_avg20=1.0,
        bb_upper=102.0, bb_lower=98.0, bb_mid=100.0, bb_width=0.04,
        bb_width_avg20=0.05, bb_squeeze=False,
        obv=10000.0, obv_slope=-0.01, rvol=1.0, vwap=100.5,
        resistance_levels=[105.0], support_levels=[95.0],
        nearest_resistance=105.0, nearest_support=95.0,
        timestamp=datetime.utcnow(),
    )


def _make_misaligned_snap():
    from trading_system.data.models import MarketSnapshot

    ind = _make_bear_indicator_set()
    return MarketSnapshot(
        symbol="BTCUSDT", current_price=100.0,
        tf_15m=ind, tf_1h=ind, tf_4h=ind,
        mtf_alignment_score=0.0, spread_pct=0.001,
        liquidity_score=0.8, timestamp=datetime.utcnow(),
    )


def _make_bull_snap():
    from trading_system.data.models import MarketSnapshot

    bull = _make_bull_indicator_set()
    tf15_bull = _make_bull_indicator_set()
    tf15_bull_sq = _make_bull_indicator_set()

    return MarketSnapshot(
        symbol="BTCUSDT", current_price=100.0,
        tf_15m=_make_bull_indicator_set(),
        tf_1h=_make_bull_indicator_set(),
        tf_4h=_make_bull_indicator_set(),
        mtf_alignment_score=0.0, spread_pct=0.001,
        liquidity_score=0.8, timestamp=datetime.utcnow(),
    )


class TestMTFAlignment:
    def test_fully_misaligned_returns_low_score(self):
        from trading_system.data.multi_timeframe import compute_mtf_alignment

        snap = _make_misaligned_snap()
        score = compute_mtf_alignment(snap)
        assert score < 30.0

    def test_fully_bull_aligned_returns_high_score(self):
        from trading_system.data.multi_timeframe import compute_mtf_alignment
        from trading_system.data.models import MarketSnapshot, IndicatorSet

        ind = _make_bull_indicator_set()
        # Make ADX > 25, RVOL > 1.5, RSI > 50
        ind2 = IndicatorSet(
            ema20=ind.ema20, ema50=ind.ema50, ema200=ind.ema200,
            ema20_slope=0.01, ema50_slope=0.005, price_vs_vwap=0.002,
            rsi=60.0, rsi_prev=52.0, rsi_slope=0.02,
            macd_line=0.5, macd_signal=0.3, macd_histogram=0.2, macd_hist_slope=0.05,
            adx=30.0, di_plus=25.0, di_minus=15.0,
            atr=1.0, atr_pct=1.0, atr_avg20=1.0,
            bb_upper=102.0, bb_lower=98.0, bb_mid=100.0, bb_width=0.04,
            bb_width_avg20=0.05, bb_squeeze=True,
            obv=10000.0, obv_slope=0.01, rvol=2.0, vwap=99.5,
            resistance_levels=[105.0], support_levels=[95.0],
            nearest_resistance=105.0, nearest_support=95.0,
            timestamp=datetime.utcnow(),
        )
        snap = MarketSnapshot(
            symbol="BTCUSDT", current_price=100.0,
            tf_15m=ind2, tf_1h=ind2, tf_4h=ind2,
            mtf_alignment_score=0.0, spread_pct=0.001,
            liquidity_score=0.9, timestamp=datetime.utcnow(),
        )
        score = compute_mtf_alignment(snap)
        assert score >= 80.0

    def test_counter_trend_returns_true_for_bear(self):
        from trading_system.data.multi_timeframe import is_counter_trend

        snap = _make_misaligned_snap()
        assert is_counter_trend(snap) is True

    def test_counter_trend_returns_false_for_bull(self):
        from trading_system.data.multi_timeframe import is_counter_trend

        snap = _make_bull_snap()
        assert is_counter_trend(snap) is False


class TestDataFeedMocked:
    def test_get_snapshot_returns_market_snapshot(self):
        from unittest.mock import patch, AsyncMock
        from trading_system.data.feed import DataFeed
        from trading_system.data.models import MarketSnapshot

        df = _make_ohlcv(300)

        mock_binance = MagicMock()
        mock_binance.get_ohlcv = AsyncMock(return_value=df)
        mock_binance.get_orderbook = AsyncMock(return_value={
            "bids": [[99.5, 1000]], "asks": [[100.0, 800]], "spread_pct": 0.005
        })
        mock_alpaca = MagicMock()

        feed = DataFeed(mock_binance, mock_alpaca)

        import asyncio
        snap = asyncio.get_event_loop().run_until_complete(
            feed.get_snapshot("BTCUSDT", "crypto")
        )

        assert isinstance(snap, MarketSnapshot)
        assert snap.symbol == "BTCUSDT"
        assert snap.current_price > 0
        assert snap.tf_15m is not None
        assert snap.tf_1h is not None
        assert snap.tf_4h is not None
