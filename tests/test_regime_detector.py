"""Module 5 regime detector tests."""
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_indicator_set(
    ema20=100.0, ema50=95.0, ema200=90.0,
    adx=15.0, rsi=50.0, atr_pct=1.0, atr_avg20=1.0,
    bb_squeeze=False, bb_width=0.04, bb_width_avg20=0.05,
    rvol=1.0, price_vs_vwap=0.0,
):
    from trading_system.data.models import IndicatorSet

    return IndicatorSet(
        ema20=ema20, ema50=ema50, ema200=ema200,
        ema20_slope=0.0, ema50_slope=0.0, price_vs_vwap=price_vs_vwap,
        rsi=rsi, rsi_prev=rsi, rsi_slope=0.0,
        macd_line=0.0, macd_signal=0.0, macd_histogram=0.0, macd_hist_slope=0.0,
        adx=adx, di_plus=20.0, di_minus=10.0,
        atr=1.0, atr_pct=atr_pct, atr_avg20=atr_avg20,
        bb_upper=102.0, bb_lower=98.0, bb_mid=100.0,
        bb_width=bb_width, bb_width_avg20=bb_width_avg20, bb_squeeze=bb_squeeze,
        obv=0.0, obv_slope=0.0, rvol=rvol, vwap=100.0,
        resistance_levels=[], support_levels=[],
        nearest_resistance=105.0, nearest_support=95.0,
        timestamp=datetime.utcnow(),
    )


def _make_snap(tf4_kwargs=None, tf1_kwargs=None, tf15_kwargs=None, price=100.0):
    from trading_system.data.models import MarketSnapshot

    tf4_kwargs = tf4_kwargs or {}
    tf1_kwargs = tf1_kwargs or {}
    tf15_kwargs = tf15_kwargs or {}

    return MarketSnapshot(
        symbol="BTCUSDT",
        current_price=price,
        tf_4h=_make_indicator_set(**tf4_kwargs),
        tf_1h=_make_indicator_set(**tf1_kwargs),
        tf_15m=_make_indicator_set(**tf15_kwargs),
        mtf_alignment_score=0.0, spread_pct=0.001,
        liquidity_score=0.8, timestamp=datetime.utcnow(),
    )


class TestRegimeDetector:
    def test_trending_bull(self):
        from trading_system.agents.regime_detector import RegimeDetector
        from trading_system.config.constants import Regime

        snap = _make_snap(
            tf4_kwargs=dict(adx=30.0, ema20=100.0, ema50=95.0, ema200=90.0, rsi=60.0),
            tf1_kwargs=dict(rsi=60.0),
            price=100.0,
        )
        detector = RegimeDetector()
        result = detector.detect(snap)
        assert result.regime == Regime.TRENDING_BULL

    def test_ranging(self):
        from trading_system.agents.regime_detector import RegimeDetector
        from trading_system.config.constants import Regime

        snap = _make_snap(
            tf4_kwargs=dict(adx=12.0, bb_width=0.03, bb_width_avg20=0.05),
            tf15_kwargs=dict(rvol=0.8),
        )
        detector = RegimeDetector()
        result = detector.detect(snap)
        assert result.regime == Regime.RANGING

    def test_high_vol_chaos_overrides_bull(self):
        from trading_system.agents.regime_detector import RegimeDetector
        from trading_system.config.constants import Regime

        # Bull indicators but 15m ATR = 3x avg
        snap = _make_snap(
            tf4_kwargs=dict(adx=30.0, ema20=100.0, ema50=95.0, ema200=90.0),
            tf15_kwargs=dict(atr_pct=3.0, atr_avg20=1.0),
        )
        detector = RegimeDetector()
        result = detector.detect(snap)
        assert result.regime == Regime.HIGH_VOL_CHAOS

    def test_vol_compression(self):
        from trading_system.agents.regime_detector import RegimeDetector
        from trading_system.config.constants import Regime

        snap = _make_snap(
            tf4_kwargs=dict(adx=15.0, atr_pct=0.7, atr_avg20=1.0),
            tf15_kwargs=dict(bb_squeeze=True),
        )
        detector = RegimeDetector()
        result = detector.detect(snap)
        assert result.regime == Regime.VOL_COMPRESSION

    def test_regime_stability_score(self):
        from trading_system.agents.regime_detector import RegimeDetector
        from trading_system.agents.regime_detector import RegimeHistory
        from trading_system.config.constants import Regime
        import tempfile
        import os

        snap = _make_snap(
            tf4_kwargs=dict(adx=12.0, bb_width=0.03, bb_width_avg20=0.05),
            tf15_kwargs=dict(rvol=0.8),
        )
        detector = RegimeDetector()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            history = RegimeHistory(db_path=db_path)
            for _ in range(15):
                result = detector.detect(snap)
                history.record(result)

            stability = history.get_stability_score()
            assert stability >= 0.6
        finally:
            os.unlink(db_path)

    def test_trending_bear_has_no_strategies(self):
        from trading_system.agents.regime_detector import RegimeDetector
        from trading_system.config.constants import Regime

        snap = _make_snap(
            tf4_kwargs=dict(adx=30.0, ema20=90.0, ema50=95.0, ema200=100.0, rsi=40.0),
            tf1_kwargs=dict(rsi=40.0),
            price=88.0,
        )
        detector = RegimeDetector()
        result = detector.detect(snap)
        assert result.regime == Regime.TRENDING_BEAR
        assert result.allowed_strategies == []
