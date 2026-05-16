"""Bar-by-bar backtest engine with no lookahead bias."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from trading_system.backtest.metrics import MetricsResult, compute_metrics
from trading_system.config.constants import Direction, Regime
from trading_system.config.settings import settings
from trading_system.core.logger import get_logger
from trading_system.data.feed import _empty_indicator_set
from trading_system.data.models import MarketSnapshot

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Full result from a backtest run."""
    metrics: MetricsResult
    trades: list[dict]
    equity_curve: list[float]
    initial_capital: float
    final_equity: float
    start_date: datetime
    end_date: datetime


def _build_snap_from_row(
    symbol: str,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    idx_15m: int,
    ts: datetime,
) -> MarketSnapshot:
    """Build a MarketSnapshot from bar-aligned DataFrames (no lookahead)."""
    sub_15m = df_15m.iloc[: idx_15m + 1]

    # Find aligned 1h and 4h bar indices
    ts_now = df_15m.index[idx_15m]
    idx_1h = df_1h.index.searchsorted(ts_now, side="right") - 1
    idx_4h = df_4h.index.searchsorted(ts_now, side="right") - 1
    idx_1h = max(idx_1h, 0)
    idx_4h = max(idx_4h, 0)

    sub_1h = df_1h.iloc[: idx_1h + 1]
    sub_4h = df_4h.iloc[: idx_4h + 1]

    try:
        from trading_system.data.indicators import compute_indicator_set
        ind_15m = compute_indicator_set(sub_15m, ts)
    except Exception:
        ind_15m = _empty_indicator_set(sub_15m, ts)

    try:
        from trading_system.data.indicators import compute_indicator_set
        ind_1h = compute_indicator_set(sub_1h, ts)
    except Exception:
        ind_1h = _empty_indicator_set(sub_1h, ts)

    try:
        from trading_system.data.indicators import compute_indicator_set
        ind_4h = compute_indicator_set(sub_4h, ts)
    except Exception:
        ind_4h = _empty_indicator_set(sub_4h, ts)

    current_price = float(df_15m["close"].iloc[idx_15m])

    from trading_system.data.multi_timeframe import compute_mtf_alignment
    snap = MarketSnapshot(
        symbol=symbol,
        current_price=current_price,
        tf_15m=ind_15m,
        tf_1h=ind_1h,
        tf_4h=ind_4h,
        mtf_alignment_score=0.0,
        spread_pct=0.001,
        liquidity_score=0.8,
        timestamp=ts,
    )
    return MarketSnapshot(
        symbol=snap.symbol, current_price=snap.current_price,
        tf_15m=snap.tf_15m, tf_1h=snap.tf_1h, tf_4h=snap.tf_4h,
        mtf_alignment_score=compute_mtf_alignment(snap),
        spread_pct=snap.spread_pct, liquidity_score=snap.liquidity_score,
        timestamp=snap.timestamp,
    )


class BacktestEngine:
    """Realistic bar-by-bar backtester. LONG only, no lookahead."""

    def run(
        self,
        ohlcv_data: dict[str, dict[str, pd.DataFrame]],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 10.0,
        slippage_pct: float = 0.0005,
        commission_pct: float = 0.001,
    ) -> BacktestResult:
        """
        ohlcv_data: {symbol: {"15m": df, "1h": df, "4h": df}}
        Iterates 15m bars as simulation clock.
        """
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
        from trading_system.risk.position_sizer import PositionSizer
        from trading_system.risk.trade_manager import TradeManager

        regime_detector = RegimeDetector()
        agents = {
            "TrendAgent": TrendAgent(),
            "MomentumAgent": MomentumAgent(),
            "VolumeAgent": VolumeAgent(),
            "VolatilityAgent": VolatilityAgent(),
            "LiquidityAgent": LiquidityAgent(),
            "SentimentAgent": SentimentAgent(),
            "ExecutionQualityAgent": ExecutionQualityAgent(),
            "EarlyMomentumAgent": EarlyMomentumAgent(),
        }
        confirmation_agent = ConfirmationAgent()
        ensemble_engine = EnsembleEngine()
        trade_manager = TradeManager()
        sizer = PositionSizer()

        equity = initial_capital
        equity_curve: list[float] = [equity]
        all_trades: list[dict] = []
        open_positions: list[dict] = []

        # Use first symbol's 15m bars as clock
        primary_symbol = next(iter(ohlcv_data))
        df_15m_primary = ohlcv_data[primary_symbol]["15m"]
        if "timestamp" in df_15m_primary.columns:
            df_15m_primary = df_15m_primary.set_index("timestamp")

        bars = df_15m_primary[(df_15m_primary.index >= pd.Timestamp(start_date)) &
                               (df_15m_primary.index <= pd.Timestamp(end_date))]

        logger.info("Backtest starting",
                    extra={"symbol": primary_symbol, "bars": len(bars),
                           "capital": initial_capital})

        for i, (ts, _) in enumerate(bars.iterrows()):
            if i < 250:  # Need 250 bars for indicators
                equity_curve.append(equity)
                continue

            # Update open positions first
            updated_positions = []
            for pos in open_positions:
                symbol = pos["symbol"]
                dfs = ohlcv_data.get(symbol)
                if not dfs:
                    updated_positions.append(pos)
                    continue

                df15 = dfs["15m"]
                if "timestamp" in df15.columns:
                    df15 = df15.set_index("timestamp")
                df1 = dfs["1h"]
                if "timestamp" in df1.columns:
                    df1 = df1.set_index("timestamp")
                df4 = dfs["4h"]
                if "timestamp" in df4.columns:
                    df4 = df4.set_index("timestamp")

                idx = df15.index.searchsorted(ts, side="right") - 1
                if idx < 0:
                    updated_positions.append(pos)
                    continue

                snap = _build_snap_from_row(symbol, df15, df1, df4, idx, ts)
                updated_pos = trade_manager.update_position(pos, snap)

                if "exit_signal" in updated_pos:
                    exit_price = snap.current_price * (1 - slippage_pct)
                    pnl_dollar = (exit_price - updated_pos["entry_price"]) * updated_pos["quantity"]
                    pnl_dollar -= updated_pos["quantity"] * exit_price * commission_pct
                    pnl_pct = (exit_price - updated_pos["entry_price"]) / updated_pos["entry_price"]
                    equity += pnl_dollar
                    all_trades.append({
                        **updated_pos,
                        "exit_price": exit_price,
                        "exit_time": ts,
                        "pnl_dollar": pnl_dollar,
                        "pnl_pct": pnl_pct,
                        "duration_bars": i - updated_pos.get("entry_bar", i),
                        "regime": updated_pos.get("regime", "unknown"),
                        "strategy_type": updated_pos.get("strategy_type", "standard"),
                        "mae": updated_pos.get("mae", 0.0),
                        "mfe": updated_pos.get("mfe", 0.0),
                    })
                else:
                    # Update MAE/MFE
                    entry = updated_pos["entry_price"]
                    cur = snap.current_price
                    move = (cur - entry) / entry
                    updated_pos["mae"] = min(updated_pos.get("mae", 0.0), move)
                    updated_pos["mfe"] = max(updated_pos.get("mfe", 0.0), move)
                    updated_positions.append(updated_pos)

            open_positions = updated_positions

            # Scan for new entries (max concurrent check)
            if len(open_positions) < settings.MAX_CONCURRENT_TRADES:
                for symbol, dfs in ohlcv_data.items():
                    if any(p["symbol"] == symbol for p in open_positions):
                        continue

                    df15 = dfs["15m"]
                    if "timestamp" in df15.columns:
                        df15 = df15.set_index("timestamp")
                    df1 = dfs["1h"]
                    if "timestamp" in df1.columns:
                        df1 = df1.set_index("timestamp")
                    df4 = dfs["4h"]
                    if "timestamp" in df4.columns:
                        df4 = df4.set_index("timestamp")

                    idx = df15.index.searchsorted(ts, side="right") - 1
                    if idx < 250:
                        continue

                    snap = _build_snap_from_row(symbol, df15, df1, df4, idx, ts)
                    regime = regime_detector.detect(snap)

                    # Skip bear/chaos
                    if regime.regime in (Regime.TRENDING_BEAR, Regime.HIGH_VOL_CHAOS):
                        continue

                    agent_outputs = {}
                    for name, agent in agents.items():
                        if name == "SentimentAgent":
                            out = agent.analyze(snap, regime, fear_greed=50)
                        else:
                            out = agent.analyze(snap, regime)
                        agent_outputs[name] = out

                    confirmation = confirmation_agent.analyze(snap, regime, agent_outputs)
                    result = ensemble_engine.evaluate(agent_outputs, confirmation, regime, snap)

                    if result.trade_approved and result.direction == Direction.LONG:
                        # Next bar entry
                        if i + 1 >= len(bars):
                            continue
                        next_idx = df15.index.searchsorted(ts, side="right")
                        if next_idx >= len(df15):
                            continue
                        entry_price = float(df15["open"].iloc[next_idx]) * (1 + slippage_pct)
                        commission = entry_price * 0.001 * commission_pct

                        try:
                            snap_entry = _build_snap_from_row(symbol, df15, df1, df4, next_idx, ts)
                            sizing = sizer.calculate(snap_entry, regime, equity, result.strategy_type)
                        except Exception:
                            continue

                        cost = sizing.quantity * entry_price + commission
                        if cost > equity * 0.95:
                            continue

                        equity -= commission
                        open_positions.append({
                            "symbol": symbol,
                            "entry_price": entry_price,
                            "stop_loss": sizing.stop_loss,
                            "take_profit": sizing.take_profit,
                            "quantity": sizing.quantity,
                            "entry_time": ts,
                            "entry_bar": i,
                            "regime": regime.regime.value,
                            "strategy_type": result.strategy_type,
                            "mae": 0.0,
                            "mfe": 0.0,
                        })

            equity_curve.append(equity)

        metrics = compute_metrics(all_trades, equity_curve, initial_capital)
        return BacktestResult(
            metrics=metrics,
            trades=all_trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital,
            final_equity=equity,
            start_date=start_date,
            end_date=end_date,
        )
