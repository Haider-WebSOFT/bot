"""Module 10 risk management tests."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_indicator_set(**kwargs):
    from trading_system.data.models import IndicatorSet

    defaults = dict(
        ema20=100.0, ema50=95.0, ema200=90.0,
        ema20_slope=0.01, ema50_slope=0.005,
        price_vs_vwap=0.002, rsi=58.0, rsi_prev=52.0, rsi_slope=0.01,
        macd_line=0.1, macd_signal=0.05, macd_histogram=0.05, macd_hist_slope=0.02,
        adx=28.0, di_plus=22.0, di_minus=12.0,
        atr=1.0, atr_pct=1.0, atr_avg20=1.0,
        bb_upper=102.0, bb_lower=98.0, bb_mid=100.0,
        bb_width=0.04, bb_width_avg20=0.05, bb_squeeze=False,
        obv=10000.0, obv_slope=0.05, rvol=1.5, vwap=99.5,
        resistance_levels=[], support_levels=[],
        nearest_resistance=105.0, nearest_support=95.0,
        timestamp=datetime.utcnow(),
    )
    defaults.update(kwargs)
    return IndicatorSet(**defaults)


def _make_snap(price=100.0, atr=1.0):
    from trading_system.data.models import MarketSnapshot

    ind = _make_indicator_set(atr=atr, atr_pct=atr)
    return MarketSnapshot(
        symbol="BTCUSDT", current_price=price,
        tf_4h=ind, tf_1h=ind, tf_15m=ind,
        mtf_alignment_score=65.0, spread_pct=0.001,
        liquidity_score=0.8, timestamp=datetime.utcnow(),
    )


def _bull_regime(multiplier=1.0):
    from trading_system.agents.regime_detector import RegimeResult
    from trading_system.config.constants import Regime

    return RegimeResult(
        regime=Regime.TRENDING_BULL, confidence=0.85,
        risk_multiplier=multiplier,
        allowed_strategies=["trend_following"],
        regime_scores={}, reasoning={}, timestamp=datetime.utcnow(),
    )


def _make_portfolio_state(equity=100.0, n_open=0):
    from trading_system.risk.portfolio_risk import PortfolioState

    positions = {f"SYM{i}USDT": {"value": 5.0} for i in range(n_open)}
    return PortfolioState(
        equity=equity, peak_equity=equity,
        open_positions=positions,
        daily_pnl=0.0, daily_start_equity=equity,
        weekly_pnl=0.0, weekly_start_equity=equity,
        consecutive_losses=0,
    )


class TestPositionSizer:
    def test_stop_loss_always_below_entry(self):
        from trading_system.risk.position_sizer import PositionSizer

        snap = _make_snap(price=100.0, atr=1.5)
        regime = _bull_regime()
        sizer = PositionSizer()
        result = sizer.calculate(snap, regime, account_equity=10.0)

        assert result.stop_loss < result.entry_price

    def test_take_profit_always_above_entry(self):
        from trading_system.risk.position_sizer import PositionSizer

        snap = _make_snap(price=50000.0, atr=500.0)
        regime = _bull_regime()
        sizer = PositionSizer()
        result = sizer.calculate(snap, regime, account_equity=10.0)

        assert result.take_profit > result.entry_price

    def test_small_account_produces_valid_size(self):
        from trading_system.risk.position_sizer import PositionSizer

        snap = _make_snap(price=100.0, atr=1.0)
        regime = _bull_regime()
        sizer = PositionSizer()
        result = sizer.calculate(snap, regime, account_equity=7.0)

        assert result.quantity > 0
        assert result.direction.value == "long"

    def test_direction_always_long(self):
        from trading_system.risk.position_sizer import PositionSizer
        from trading_system.config.constants import Direction

        snap = _make_snap()
        regime = _bull_regime()
        sizer = PositionSizer()
        result = sizer.calculate(snap, regime, account_equity=10.0)

        assert result.direction == Direction.LONG


class TestPortfolioRiskEngine:
    def test_daily_loss_3pct_rejects_trade(self):
        from trading_system.risk.portfolio_risk import PortfolioRiskEngine, PortfolioState
        from trading_system.risk.position_sizer import SizingResult
        from trading_system.config.constants import Direction

        state = PortfolioState(
            equity=97.0, peak_equity=100.0,
            open_positions={},
            daily_pnl=-3.0, daily_start_equity=100.0,
            weekly_pnl=-3.0, weekly_start_equity=100.0,
            consecutive_losses=0,
        )
        sizing = SizingResult(
            symbol="BTCUSDT", direction=Direction.LONG,
            quantity=0.001, entry_price=50000.0,
            stop_loss=49000.0, take_profit=52000.0,
            risk_dollars=1.0, r_r_ratio=2.0, atr_used=500.0,
            regime_multiplier=1.0, size_reduction_reason=None,
        )
        engine = PortfolioRiskEngine()
        allowed, reason = engine.can_open_trade(state, sizing, "BTCUSDT")
        assert not allowed

    def test_max_concurrent_trades_rejects_7th(self):
        from trading_system.risk.portfolio_risk import PortfolioRiskEngine
        from trading_system.risk.position_sizer import SizingResult
        from trading_system.config.constants import Direction

        state = _make_portfolio_state(equity=100.0, n_open=6)
        sizing = SizingResult(
            symbol="NEWUSDT", direction=Direction.LONG,
            quantity=0.1, entry_price=10.0,
            stop_loss=9.0, take_profit=12.0,
            risk_dollars=0.5, r_r_ratio=2.0, atr_used=0.5,
            regime_multiplier=1.0, size_reduction_reason=None,
        )
        engine = PortfolioRiskEngine()
        allowed, reason = engine.can_open_trade(state, sizing, "NEWUSDT")
        assert not allowed
        assert "Max concurrent" in reason


class TestTradeManager:
    def test_pnl_r_1_moves_stop_to_breakeven(self):
        from trading_system.risk.trade_manager import TradeManager

        # ATR=1, entry=100, SL=98.5 (1.5*ATR), so at 1R, price=100+(100-98.5)=101.5
        snap = _make_snap(price=101.5, atr=1.0)
        position = {
            "symbol": "BTCUSDT",
            "entry_price": 100.0,
            "stop_loss": 98.5,
            "take_profit": 104.5,
            "entry_time": datetime.utcnow() - timedelta(hours=1),
        }
        mgr = TradeManager()
        updated = mgr.update_position(position, snap)

        assert updated["stop_loss"] > 100.0  # above entry = break-even
        assert updated.get("breakeven_moved") is True

    def test_stop_loss_triggered_when_price_at_stop(self):
        from trading_system.risk.trade_manager import TradeManager
        from trading_system.config.constants import ExitReason

        snap = _make_snap(price=97.0, atr=1.0)
        position = {
            "symbol": "BTCUSDT",
            "entry_price": 100.0,
            "stop_loss": 98.0,
            "take_profit": 105.0,
            "entry_time": datetime.utcnow() - timedelta(hours=1),
        }
        mgr = TradeManager()
        updated = mgr.update_position(position, snap)

        assert updated.get("exit_signal") == ExitReason.STOP_LOSS
