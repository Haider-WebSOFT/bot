"""Module 7 agent tests."""
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
    atr=1.0, bb_lower=98.0,
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
        bb_upper=102.0, bb_lower=bb_lower, bb_mid=100.0,
        bb_width=bb_width, bb_width_avg20=bb_width_avg20,
        bb_squeeze=bb_squeeze,
        obv=obv, obv_slope=obv_slope, rvol=rvol, vwap=vwap,
        resistance_levels=[], support_levels=[],
        nearest_resistance=105.0, nearest_support=95.0,
        timestamp=datetime.utcnow(),
    )


def _make_snap(tf4=None, tf1=None, tf15=None, price=100.0,
               symbol="BTCUSDT", liquidity_score=0.8, spread_pct=0.001):
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


def _ranging_regime():
    from trading_system.agents.regime_detector import RegimeResult
    from trading_system.config.constants import Regime

    return RegimeResult(
        regime=Regime.RANGING, confidence=0.75,
        risk_multiplier=0.7,
        allowed_strategies=["mean_reversion"],
        regime_scores={}, reasoning={},
        timestamp=datetime.utcnow(),
    )


class TestLiquidityAgent:
    def test_wide_spread_hard_block_scores_zero(self):
        from trading_system.agents.liquidity_agent import LiquidityAgent

        snap = _make_snap(spread_pct=0.006, liquidity_score=0.8)
        agent = LiquidityAgent()
        out = agent.analyze(snap, _bull_regime())
        assert out.score == 0
        assert "hard_block" in out.reasoning

    def test_btcusdt_good_depth_tight_spread_scores_high(self):
        from trading_system.agents.liquidity_agent import LiquidityAgent

        snap = _make_snap(symbol="BTCUSDT", spread_pct=0.0005,
                          liquidity_score=0.9)
        agent = LiquidityAgent()
        out = agent.analyze(snap, _bull_regime())
        assert out.score > 60

    def test_direction_always_long_or_none(self):
        from trading_system.agents.liquidity_agent import LiquidityAgent
        from trading_system.config.constants import Direction

        agent = LiquidityAgent()
        snap = _make_snap()
        out = agent.analyze(snap, _bull_regime())
        assert out.direction in (Direction.LONG, None)


class TestExecutionQualityAgent:
    def test_slippage_3x_expected_drops_score(self):
        from trading_system.agents.execution_quality_agent import ExecutionQualityAgent

        agent = ExecutionQualityAgent()
        for _ in range(10):
            agent.record_execution(100.0, 100.15, True, 50.0)  # 0.15% vs 0.05%

        snap = _make_snap()
        out = agent.analyze(snap, _bull_regime())
        assert out.score < 50

    def test_direction_always_long_or_none(self):
        from trading_system.agents.execution_quality_agent import ExecutionQualityAgent
        from trading_system.config.constants import Direction

        agent = ExecutionQualityAgent()
        for _ in range(10):
            agent.record_execution(100.0, 100.05, True, 40.0)

        snap = _make_snap()
        out = agent.analyze(snap, _bull_regime())
        assert out.direction in (Direction.LONG, None)


class TestSentimentAgent:
    def test_extreme_fear_scores_high(self):
        from trading_system.agents.sentiment_agent import SentimentAgent

        agent = SentimentAgent()
        snap = _make_snap()
        out = agent.analyze(snap, _bull_regime(), fear_greed=15, vix=18.0)
        assert out.score > 60

    def test_direction_always_long_or_none(self):
        from trading_system.agents.sentiment_agent import SentimentAgent
        from trading_system.config.constants import Direction

        agent = SentimentAgent()
        snap = _make_snap()
        out = agent.analyze(snap, _bull_regime(), fear_greed=50, vix=18.0)
        assert out.direction in (Direction.LONG, None)


class TestVolatilityAgent:
    def test_atr_3x_avg_blocks_entry(self):
        from trading_system.agents.volatility_agent import VolatilityAgent

        tf15 = _make_indicator_set(atr_pct=3.0, atr_avg20=1.0)
        snap = _make_snap(tf15=tf15)
        agent = VolatilityAgent()
        out = agent.analyze(snap, _bull_regime())
        assert out.score == 0

    def test_bb_squeeze_releasing_scores_high(self):
        from trading_system.agents.volatility_agent import VolatilityAgent

        tf15 = _make_indicator_set(
            bb_squeeze=True, bb_width=0.046, bb_width_avg20=0.05,
            atr_pct=1.0, atr_avg20=1.0,
        )
        snap = _make_snap(tf15=tf15)
        agent = VolatilityAgent()
        out = agent.analyze(snap, _bull_regime())
        assert out.score >= 40

    def test_direction_always_long_or_none(self):
        from trading_system.agents.volatility_agent import VolatilityAgent
        from trading_system.config.constants import Direction

        agent = VolatilityAgent()
        snap = _make_snap()
        out = agent.analyze(snap, _bull_regime())
        assert out.direction in (Direction.LONG, None)
