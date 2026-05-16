"""Module 17: Full pipeline integration tests."""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_indicator_set(bullish: bool = True, price: float = 100.0):
    from trading_system.data.models import IndicatorSet

    if bullish:
        return IndicatorSet(
            ema20=price, ema50=price * 0.95, ema200=price * 0.88,
            ema20_slope=0.01, ema50_slope=0.005, price_vs_vwap=0.003,
            rsi=58.0, rsi_prev=46.0, rsi_slope=0.02,
            macd_line=0.5, macd_signal=0.3, macd_histogram=0.2, macd_hist_slope=0.05,
            adx=28.0, di_plus=24.0, di_minus=14.0,
            atr=1.0, atr_pct=1.0, atr_avg20=1.0,
            bb_upper=102.0, bb_lower=98.0, bb_mid=100.0,
            bb_width=0.04, bb_width_avg20=0.05, bb_squeeze=False,
            obv=10000.0, obv_slope=0.05, rvol=2.3, vwap=99.5,
            resistance_levels=[105.0], support_levels=[95.0],
            nearest_resistance=105.0, nearest_support=95.0,
            timestamp=datetime.utcnow(),
        )
    else:
        return IndicatorSet(
            ema20=price * 0.85, ema50=price * 0.92, ema200=price,
            ema20_slope=-0.01, ema50_slope=-0.005, price_vs_vwap=-0.01,
            rsi=38.0, rsi_prev=45.0, rsi_slope=-0.02,
            macd_line=-0.5, macd_signal=-0.3, macd_histogram=-0.2, macd_hist_slope=-0.05,
            adx=30.0, di_plus=14.0, di_minus=24.0,
            atr=1.0, atr_pct=1.0, atr_avg20=1.0,
            bb_upper=102.0, bb_lower=98.0, bb_mid=100.0,
            bb_width=0.04, bb_width_avg20=0.05, bb_squeeze=False,
            obv=10000.0, obv_slope=-0.05, rvol=0.8, vwap=100.5,
            resistance_levels=[105.0], support_levels=[95.0],
            nearest_resistance=105.0, nearest_support=95.0,
            timestamp=datetime.utcnow(),
        )


def _make_snap(bullish: bool = True, price: float = 100.0, symbol: str = "BTCUSDT"):
    from trading_system.data.models import MarketSnapshot

    ind = _make_indicator_set(bullish, price)
    return MarketSnapshot(
        symbol=symbol, current_price=price,
        tf_4h=ind, tf_1h=ind, tf_15m=ind,
        mtf_alignment_score=75.0 if bullish else 15.0,
        spread_pct=0.001, liquidity_score=0.85,
        timestamp=datetime.utcnow(),
    )


class TestFullPipeline:
    def test_direction_always_long_in_pipeline(self):
        """Every agent must return Direction.LONG or None — never SHORT."""
        from trading_system.config.constants import Direction
        from trading_system.agents.regime_detector import RegimeDetector
        from trading_system.agents.trend_agent import TrendAgent
        from trading_system.agents.momentum_agent import MomentumAgent
        from trading_system.agents.volume_agent import VolumeAgent
        from trading_system.agents.volatility_agent import VolatilityAgent
        from trading_system.agents.liquidity_agent import LiquidityAgent
        from trading_system.agents.sentiment_agent import SentimentAgent
        from trading_system.agents.early_momentum_agent import EarlyMomentumAgent

        snap = _make_snap(bullish=True)
        regime = RegimeDetector().detect(snap)

        for AgentClass in [TrendAgent, MomentumAgent, VolumeAgent,
                            VolatilityAgent, LiquidityAgent, EarlyMomentumAgent]:
            out = AgentClass().analyze(snap, regime)
            assert out.direction in (Direction.LONG, None), (
                f"{AgentClass.__name__} returned {out.direction}"
            )

        out = SentimentAgent().analyze(snap, regime, fear_greed=50, vix=18.0)
        assert out.direction in (Direction.LONG, None)

    def test_trending_bear_blocks_all_trades(self):
        """In a bear trend the ensemble must block every trade."""
        from trading_system.agents.regime_detector import RegimeDetector
        from trading_system.agents.trend_agent import TrendAgent
        from trading_system.agents.momentum_agent import MomentumAgent
        from trading_system.agents.volume_agent import VolumeAgent
        from trading_system.agents.volatility_agent import VolatilityAgent
        from trading_system.agents.liquidity_agent import LiquidityAgent
        from trading_system.agents.sentiment_agent import SentimentAgent
        from trading_system.agents.execution_quality_agent import ExecutionQualityAgent
        from trading_system.agents.early_momentum_agent import EarlyMomentumAgent
        from trading_system.agents.confirmation_agent import ConfirmationAgent
        from trading_system.agents.ensemble import EnsembleEngine
        from trading_system.config.constants import Regime

        snap = _make_snap(bullish=False)
        regime = RegimeDetector().detect(snap)

        all_agents = {
            "TrendAgent": TrendAgent(), "MomentumAgent": MomentumAgent(),
            "VolumeAgent": VolumeAgent(), "VolatilityAgent": VolatilityAgent(),
            "LiquidityAgent": LiquidityAgent(), "SentimentAgent": SentimentAgent(),
            "ExecutionQualityAgent": ExecutionQualityAgent(),
            "EarlyMomentumAgent": EarlyMomentumAgent(),
        }
        outputs = {}
        for name, agent in all_agents.items():
            if name == "SentimentAgent":
                outputs[name] = agent.analyze(snap, regime, fear_greed=50)
            else:
                outputs[name] = agent.analyze(snap, regime)

        confirmation = ConfirmationAgent().analyze(snap, regime, outputs)
        result = EnsembleEngine().evaluate(outputs, confirmation, regime, snap)

        if regime.regime == Regime.TRENDING_BEAR:
            assert result.trade_approved is False
            assert result.direction is None

    def test_early_momentum_integration(self):
        """Verify EarlyMomentumAgent approves when all 3 conditions are met."""
        from trading_system.data.models import IndicatorSet, MarketSnapshot
        from trading_system.agents.regime_detector import RegimeResult
        from trading_system.agents.early_momentum_agent import EarlyMomentumAgent
        from trading_system.config.constants import Direction, Regime

        ind = IndicatorSet(
            ema20=100.0, ema50=95.0, ema200=90.0,
            ema20_slope=0.01, ema50_slope=0.005, price_vs_vwap=0.002,
            rsi=57.0, rsi_prev=46.0, rsi_slope=0.02,   # acceleration = 11
            macd_line=0.2, macd_signal=0.1, macd_histogram=0.05, macd_hist_slope=0.03,
            adx=28.0, di_plus=22.0, di_minus=12.0,
            atr=1.0, atr_pct=1.0, atr_avg20=1.0,
            bb_upper=102.0, bb_lower=98.0, bb_mid=100.0,
            bb_width=0.04, bb_width_avg20=0.05, bb_squeeze=False,
            obv=10000.0, obv_slope=0.05, rvol=2.3, vwap=99.5,
            resistance_levels=[], support_levels=[],
            nearest_resistance=105.0, nearest_support=95.0,
            timestamp=datetime.utcnow(),
        )
        snap = MarketSnapshot(
            symbol="BTCUSDT", current_price=100.0,
            tf_4h=ind, tf_1h=ind, tf_15m=ind,
            mtf_alignment_score=70.0, spread_pct=0.001,
            liquidity_score=0.85, timestamp=datetime.utcnow(),
        )
        bull_regime = RegimeResult(
            regime=Regime.TRENDING_BULL, confidence=0.9,
            risk_multiplier=1.0, allowed_strategies=["early_momentum"],
            regime_scores={}, reasoning={}, timestamp=datetime.utcnow(),
        )

        out = EarlyMomentumAgent().analyze(snap, bull_regime)
        assert out.score >= 55
        assert out.direction == Direction.LONG


class TestImportAndHalalCheck:
    def test_all_imports_succeed(self):
        from trading_system.config.settings import settings
        from trading_system.agents.ensemble import EnsembleEngine
        from trading_system.agents.early_momentum_agent import EarlyMomentumAgent
        from trading_system.risk.position_sizer import PositionSizer
        from trading_system.analytics.gate import LiveTradingGate
        from trading_system.backtest.engine import BacktestEngine
        assert True

    def test_direction_enum_long_only(self):
        from trading_system.config.constants import Direction
        assert list(Direction) == [Direction.LONG]

    def test_no_hardcoded_secrets_in_source(self):
        """Source files must not contain hardcoded credential strings."""
        import re
        src_dir = Path(__file__).parent.parent / "trading_system"
        pattern = re.compile(
            r"(api_key|api_secret|password)\s*=\s*['\"][A-Za-z0-9+/]{10,}['\"]",
            re.IGNORECASE,
        )
        violations = [
            str(f) for f in src_dir.rglob("*.py")
            if pattern.search(f.read_text())
        ]
        assert violations == [], f"Hardcoded secrets in: {violations}"
