"""Manages open LONG positions: stops, trailing, partial exits, time exits."""
from __future__ import annotations

from datetime import datetime, timedelta

from trading_system.config.constants import ExitReason
from trading_system.config.settings import settings
from trading_system.core.logger import get_logger
from trading_system.data.models import MarketSnapshot

logger = get_logger(__name__)


class TradeManager:
    """Manages open LONG positions after entry. Called on each bar close."""

    def update_position(self, position: dict, snap: MarketSnapshot) -> dict:
        """
        Apply position management rules to a single LONG position.
        Returns updated position dict with exit_signal if triggered.
        """
        pos = dict(position)
        current_price = snap.current_price
        entry_price = pos["entry_price"]
        stop_loss = pos["stop_loss"]
        take_profit = pos["take_profit"]
        atr = snap.tf_1h.atr
        entry_time: datetime = pos.get("entry_time", datetime.utcnow())

        sl_distance = entry_price - stop_loss
        pnl_r = (current_price - entry_price) / max(sl_distance, 1e-9)
        pos["pnl_r"] = pnl_r

        # 1. Break-even: move SL above entry at 1R
        if pnl_r >= settings.ATR_BREAKEVEN_TRIGGER and not pos.get("breakeven_moved"):
            new_sl = entry_price + (0.1 * atr)
            if new_sl > stop_loss:
                pos["stop_loss"] = new_sl
                pos["breakeven_moved"] = True
                logger.info("Break-even stop moved",
                            extra={"symbol": pos["symbol"], "new_sl": new_sl})

        # 2. Trailing stop at 2R
        if pnl_r >= 2.0:
            trail = current_price - (atr * settings.ATR_TRAILING_MULTIPLIER)
            if trail > pos.get("stop_loss", stop_loss):
                pos["stop_loss"] = trail
                logger.debug("Trailing stop updated",
                             extra={"symbol": pos["symbol"], "trail": trail})

        # 3. Partial profit at 1.5R
        if pnl_r >= 1.5 and not pos.get("partially_exited"):
            pos["exit_signal"] = ExitReason.PARTIAL_EXIT
            pos["exit_qty_pct"] = 0.50
            pos["partially_exited"] = True

        # 4. Time exit: > 48 hours
        hours_open = (datetime.utcnow() - entry_time).total_seconds() / 3600
        if hours_open > 48 and not pos.get("exit_signal"):
            pos["exit_signal"] = ExitReason.TIME_EXIT

        # 5. Momentum exit: RSI crosses below 50 and MACD hist negative
        tf1 = snap.tf_1h
        if (tf1.rsi < 50 and tf1.rsi_prev >= 50 and tf1.macd_histogram < 0
                and not pos.get("exit_signal")):
            pos["exit_signal"] = ExitReason.MOMENTUM_EXIT

        # 6. Stop loss
        if current_price <= pos["stop_loss"] and not pos.get("exit_signal"):
            pos["exit_signal"] = ExitReason.STOP_LOSS

        # 7. Take profit
        if current_price >= take_profit and not pos.get("exit_signal"):
            pos["exit_signal"] = ExitReason.TAKE_PROFIT

        return pos
