"""Module 13 paper trading tests."""
import sys
import tempfile
import os
from datetime import datetime
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


def _make_snap(price=100.0, rvol=1.5, spread_pct=0.001):
    from trading_system.data.models import MarketSnapshot

    ind = _make_indicator_set(rvol=rvol)
    return MarketSnapshot(
        symbol="BTCUSDT", current_price=price,
        tf_4h=ind, tf_1h=ind, tf_15m=ind,
        mtf_alignment_score=65.0, spread_pct=spread_pct,
        liquidity_score=0.8, timestamp=datetime.utcnow(),
    )


def _make_card(strategy_type="standard"):
    from trading_system.scanner.ranker import OpportunityCard
    from trading_system.config.constants import Direction, Regime

    return OpportunityCard(
        symbol="BTCUSDT", asset_type="crypto",
        direction=Direction.LONG,
        strategy_type=strategy_type,
        regime=Regime.TRENDING_BULL,
        regime_confidence=0.85,
        ensemble_score=70.0,
        agent_scores={},
        suggested_entry=100.0,
        suggested_sl=98.5,
        suggested_tp=104.5,
        r_r_ratio=2.0,
        ev_score=50.0,
        liquidity_score=0.8,
        rvol=2.0,
        mtf_alignment=70.0,
        reasoning_summary="test",
        timestamp=datetime.utcnow(),
    )


def _make_sizing(qty=0.1, entry=100.0):
    from trading_system.risk.position_sizer import SizingResult
    from trading_system.config.constants import Direction

    return SizingResult(
        symbol="BTCUSDT", direction=Direction.LONG,
        quantity=qty, entry_price=entry,
        stop_loss=entry - 1.5, take_profit=entry + 3.0,
        risk_dollars=0.10, r_r_ratio=2.0, atr_used=1.0,
        regime_multiplier=1.0, size_reduction_reason=None,
    )


class TestSlippage:
    def test_high_rvol_increases_slippage(self):
        from trading_system.config.settings import settings

        def calc_slip(snap):
            s = settings.SLIPPAGE_PCT
            if snap.tf_15m.rvol > 2.0:
                s *= 1.5
            if snap.spread_pct > 0.001:
                s += snap.spread_pct / 2
            return s

        snap_high = _make_snap(rvol=2.5)
        snap_low = _make_snap(rvol=1.0)
        assert calc_slip(snap_high) > calc_slip(snap_low)


class TestTradeJournal:
    def test_open_close_trade_saves_correctly(self):
        from trading_system.paper_trading.trade_journal import TradeJournal

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            journal = TradeJournal(db_path=db_path)
            card = _make_card("standard")
            sizing = _make_sizing()

            trade_id = journal.open_trade(card, sizing, actual_entry_price=100.05)
            assert trade_id is not None and trade_id > 0

            journal.close_trade(trade_id, 103.0, "take_profit", mae=-0.005, mfe=0.03)

            trades = journal.get_all_trades()
            assert len(trades) == 1
            t = trades[0]
            assert t["direction"] == "long"
            assert t["strategy_type"] == "standard"
            assert t["exit_reason"] == "take_profit"
            assert t["exit_price"] == pytest.approx(103.0)
        finally:
            os.unlink(db_path)

    def test_early_momentum_strategy_type_saved(self):
        from trading_system.paper_trading.trade_journal import TradeJournal

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            journal = TradeJournal(db_path=db_path)
            card = _make_card("early_momentum")
            sizing = _make_sizing()

            trade_id = journal.open_trade(card, sizing, actual_entry_price=100.05)
            journal.close_trade(trade_id, 102.0, "take_profit", 0.0, 0.02)

            trades = journal.get_all_trades()
            assert trades[0]["strategy_type"] == "early_momentum"
        finally:
            os.unlink(db_path)

    def test_direction_always_long(self):
        from trading_system.paper_trading.trade_journal import TradeJournal

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            journal = TradeJournal(db_path=db_path)
            journal.open_trade(_make_card(), _make_sizing(), 100.0)
            trades = journal.get_all_trades()
            assert all(t["direction"] == "long" for t in trades)
        finally:
            os.unlink(db_path)
