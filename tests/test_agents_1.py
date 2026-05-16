"""Module 6 agent tests."""
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_indicator_set(
    ema20=100.0, ema50=95.0, ema200=90.0,
    ema20_slope=0.01, ema50_slope=0.005,
    adx=15.0, rsi=50.0, rsi_prev=48.0, rsi_slope=0.0,
    price_vs_vwap=0.0,
    macd_histogram=0.0, macd_hist_slope=0.0,
    macd_line=0.0, macd_signal=0.0,
    atr_pct=1.0, atr_avg20=1.0,
    bb_squeeze=False, bb_width=0.04, bb_width_avg20=0.05,
    rvol=1.0, obv_slope=0.0, obv=0.0, vwap=100.0,
    atr=1.0,
):
    from trading_system.data.models import IndicatorSet

    return IndicatorSet(
        ema20=ema20, ema50=ema50, ema200=ema200,
        ema20_slope=ema20_slope, ema50_slope=ema50_slope,
        price_vs_vwap=price_vs_vwap,
        rsi=rsi, rsi_prev=rsi_prev, rsi_slope=rsi_slope,
        macd_line=macd_line, macd_signal=macd_signal,
        macd_histogram=macd_histogram, macd_hist_slope=macd_hist_slope,
        adx=adx, di_plus=20.0, di_minus=10.0,
        atr=atr, atr_pct=atr_pct, atr_avg20=atr_avg20,
        bb_upper=102.0, bb_lower=98.0, bb_mid=100.0,
        bb_width=bb_width, bb_width_avg20=bb_width_avg20,
        bb_squeeze=bb_squeeze,
        obv=obv, obv_slope=obv_slope, rvol=rvol, vwap=vwap,
        resistance_levels=[], support_levels=[],
        nearest_resistance=105.0, nearest_support=95.0,
        timestamp=datetime.utcnow(),
    )


def _make_snap(tf4=None, tf1=None, tf15=None, price=100.0, symbol="BTCUSDT",
               liquidity_score=0.8, spread_pct=0.001):
    from trading_system.data.models import MarketSnapshot

    neutral = _make_indicator_set()
    return MarketSnapshot(
        symbol=symbol, current_price=price,
        tf_4h=tf4 or neutral,
        tf_1h=tf1 or neutral,
        tf_15m=tf15 or neutral,
        mtf_alignment_score=50.0,
        spread_pct=spread_pct,
        liquidity_score=liquidity_score,
        timestamp=datetime.utcnow(),
    )


def _bull_regime():
    from trading_system.agents.regime_detector import RegimeResult
    from trading_system.config.constants import Regime

    return RegimeResult(
        regime=Regime.TRENDING_BULL, confidence=0.85,
        risk_multiplier=1.0,
        allowed_strategies=["trend_following"],
        regime_scores={}, reasoning={},
        timestamp=datetime.utcnow(),
    )


class TestTrendAgent:
    def test_bull_ema_stack_adx30_vwap_above_scores_high(self):
        from trading_system.agents.trend_agent import TrendAgent

        tf4 = _make_indicator_set(ema20=100, ema50=95, ema200=90, adx=32.0)
        tf1 = _make_indicator_set(price_vs_vwap=0.005, ema20_slope=0.01)
        snap = _make_snap(tf4=tf4, tf1=tf1)

        agent = TrendAgent()
        out = agent.analyze(snap, _bull_regime())
        assert out.score > 60
        assert out.direction is not None

    def test_bear_ema_stack_scores_zero(self):
        from trading_system.agents.trend_agent import TrendAgent
        from trading_system.config.constants import Direction

        tf4 = _make_indicator_set(ema20=85, ema50=95, ema200=100, adx=10.0)
        tf1 = _make_indicator_set(price_vs_vwap=-0.01, ema20_slope=-0.01)
        snap = _make_snap(tf4=tf4, tf1=tf1)

        agent = TrendAgent()
        out = agent.analyze(snap, _bull_regime())
        assert out.score == 0.0
        assert out.direction is None

    def test_direction_always_long_or_none(self):
        from trading_system.agents.trend_agent import TrendAgent
        from trading_system.config.constants import Direction

        agent = TrendAgent()
        snap = _make_snap()
        out = agent.analyze(snap, _bull_regime())
        assert out.direction in (Direction.LONG, None)


class TestMomentumAgent:
    def test_rsi_58_rising_macd_positive_rising_scores_high(self):
        from trading_system.agents.momentum_agent import MomentumAgent

        tf1 = _make_indicator_set(rsi=58.0, rsi_slope=0.02,
                                   macd_histogram=0.3, macd_hist_slope=0.1)
        tf15 = _make_indicator_set(rsi=55.0, rsi_slope=0.01)
        snap = _make_snap(tf1=tf1, tf15=tf15)

        agent = MomentumAgent()
        out = agent.analyze(snap, _bull_regime())
        assert out.score > 50

    def test_direction_always_long_or_none(self):
        from trading_system.agents.momentum_agent import MomentumAgent
        from trading_system.config.constants import Direction

        agent = MomentumAgent()
        snap = _make_snap()
        out = agent.analyze(snap, _bull_regime())
        assert out.direction in (Direction.LONG, None)


class TestVolumeAgent:
    def test_rvol_2_5_obv_rising_scores_high(self):
        from trading_system.agents.volume_agent import VolumeAgent

        tf15 = _make_indicator_set(rvol=2.5)
        tf1 = _make_indicator_set(obv_slope=0.05, rvol=2.0,
                                   ema20=100, ema50=95)
        tf4 = _make_indicator_set(ema20=100, ema50=95)
        snap = _make_snap(tf4=tf4, tf1=tf1, tf15=tf15)

        agent = VolumeAgent()
        out = agent.analyze(snap, _bull_regime())
        assert out.score > 50

    def test_low_rvol_capped_at_20(self):
        from trading_system.agents.volume_agent import VolumeAgent

        tf15 = _make_indicator_set(rvol=0.7)
        tf1 = _make_indicator_set(obv_slope=0.1)
        snap = _make_snap(tf15=tf15, tf1=tf1)

        agent = VolumeAgent()
        out = agent.analyze(snap, _bull_regime())
        assert out.score <= 20

    def test_direction_always_long_or_none(self):
        from trading_system.agents.volume_agent import VolumeAgent
        from trading_system.config.constants import Direction

        agent = VolumeAgent()
        snap = _make_snap()
        out = agent.analyze(snap, _bull_regime())
        assert out.direction in (Direction.LONG, None)
