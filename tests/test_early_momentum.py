"""Module 8 Early Momentum Agent tests."""
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_indicator_set(**kwargs):
    from trading_system.data.models import IndicatorSet

    defaults = dict(
        ema20=100.0, ema50=95.0, ema200=90.0,
        ema20_slope=0.01, ema50_slope=0.005,
        price_vs_vwap=0.002, rsi=57.0, rsi_prev=46.0, rsi_slope=0.02,
        macd_line=0.1, macd_signal=0.05, macd_histogram=0.05, macd_hist_slope=0.02,
        adx=28.0, di_plus=22.0, di_minus=12.0,
        atr=1.0, atr_pct=1.0, atr_avg20=1.0,
        bb_upper=102.0, bb_lower=98.0, bb_mid=100.0,
        bb_width=0.04, bb_width_avg20=0.05, bb_squeeze=False,
        obv=10000.0, obv_slope=0.05, rvol=2.5, vwap=99.5,
        resistance_levels=[], support_levels=[],
        nearest_resistance=105.0, nearest_support=95.0,
        timestamp=datetime.utcnow(),
    )
    defaults.update(kwargs)
    return IndicatorSet(**defaults)


def _make_snap(tf4=None, tf1=None, tf15=None, price=100.0,
               liquidity_score=0.8, spread_pct=0.001):
    from trading_system.data.models import MarketSnapshot

    neutral = _make_indicator_set()
    return MarketSnapshot(
        symbol="BTCUSDT", current_price=price,
        tf_4h=tf4 or neutral,
        tf_1h=tf1 or neutral,
        tf_15m=tf15 or neutral,
        mtf_alignment_score=60.0, spread_pct=spread_pct,
        liquidity_score=liquidity_score, timestamp=datetime.utcnow(),
    )


def _bull_regime():
    from trading_system.agents.regime_detector import RegimeResult
    from trading_system.config.constants import Regime

    return RegimeResult(
        regime=Regime.TRENDING_BULL, confidence=0.9,
        risk_multiplier=1.0, allowed_strategies=["early_momentum"],
        regime_scores={}, reasoning={}, timestamp=datetime.utcnow(),
    )


def _ranging_regime():
    from trading_system.agents.regime_detector import RegimeResult
    from trading_system.config.constants import Regime

    return RegimeResult(
        regime=Regime.RANGING, confidence=0.75,
        risk_multiplier=0.7, allowed_strategies=["mean_reversion"],
        regime_scores={}, reasoning={}, timestamp=datetime.utcnow(),
    )


class TestEarlyMomentumAgent:
    def test_all_3_conditions_met_approves_long(self):
        from trading_system.agents.early_momentum_agent import EarlyMomentumAgent
        from trading_system.config.constants import Direction

        tf15 = _make_indicator_set(rsi=57.0, rsi_prev=46.0, rvol=2.5,
                                    obv_slope=0.05, macd_hist_slope=0.02,
                                    macd_histogram=0.05)
        tf4 = _make_indicator_set(ema20=100.0, ema50=95.0)
        snap = _make_snap(tf4=tf4, tf15=tf15)

        agent = EarlyMomentumAgent()
        out = agent.analyze(snap, _bull_regime())
        assert out.score >= 60
        assert out.direction == Direction.LONG

    def test_not_trending_bull_returns_zero(self):
        from trading_system.agents.early_momentum_agent import EarlyMomentumAgent

        tf15 = _make_indicator_set(rsi=57.0, rsi_prev=46.0, rvol=2.5, obv_slope=0.05)
        snap = _make_snap(tf15=tf15)

        agent = EarlyMomentumAgent()
        out = agent.analyze(snap, _ranging_regime())
        assert out.score == 0.0
        assert out.direction is None

    def test_low_rsi_acceleration_fails_cond1(self):
        from trading_system.agents.early_momentum_agent import EarlyMomentumAgent

        # rsi_acceleration = 57 - 52 = 5, below threshold of 8
        tf15 = _make_indicator_set(rsi=57.0, rsi_prev=52.0,
                                    rvol=2.5, obv_slope=0.05,
                                    macd_hist_slope=0.02, macd_histogram=0.05)
        tf4 = _make_indicator_set(ema20=100.0, ema50=95.0)
        snap = _make_snap(tf4=tf4, tf15=tf15)

        agent = EarlyMomentumAgent()
        out = agent.analyze(snap, _bull_regime())
        assert out.score < 55
        assert out.direction is None

    def test_low_rvol_cond2_fails(self):
        from trading_system.agents.early_momentum_agent import EarlyMomentumAgent

        # cond1 passes, cond2 fails (rvol < 2.0)
        tf15 = _make_indicator_set(rsi=57.0, rsi_prev=46.0, rvol=1.5,
                                    obv_slope=0.05, macd_hist_slope=0.02,
                                    macd_histogram=0.05)
        tf4 = _make_indicator_set(ema20=100.0, ema50=95.0)
        snap = _make_snap(tf4=tf4, tf15=tf15)

        agent = EarlyMomentumAgent()
        out = agent.analyze(snap, _bull_regime())
        # cond1(35) + cond3(10) = 45, below threshold 55
        assert out.score < 55
        assert out.direction is None

    def test_low_liquidity_caps_score(self):
        from trading_system.agents.early_momentum_agent import EarlyMomentumAgent

        tf15 = _make_indicator_set(rsi=57.0, rsi_prev=46.0, rvol=2.5,
                                    obv_slope=0.05, macd_hist_slope=0.02,
                                    macd_histogram=0.05)
        tf4 = _make_indicator_set(ema20=100.0, ema50=95.0)
        snap = _make_snap(tf4=tf4, tf15=tf15, liquidity_score=0.4)

        agent = EarlyMomentumAgent()
        out = agent.analyze(snap, _bull_regime())
        assert out.score <= 20
        assert out.direction is None

    def test_chaos_atr_blocks_all(self):
        from trading_system.agents.early_momentum_agent import EarlyMomentumAgent

        # ATR chaos override: atr_pct > atr_avg20 * 2
        tf15 = _make_indicator_set(rsi=57.0, rsi_prev=46.0, rvol=2.5,
                                    obv_slope=0.05, atr_pct=3.0, atr_avg20=1.0)
        tf4 = _make_indicator_set(ema20=100.0, ema50=95.0)
        snap = _make_snap(tf4=tf4, tf15=tf15)

        agent = EarlyMomentumAgent()
        out = agent.analyze(snap, _bull_regime())
        assert out.score == 0.0
        assert out.direction is None
