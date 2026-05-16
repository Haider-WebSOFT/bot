"""Continuous risk monitoring loop."""
from __future__ import annotations

from datetime import datetime

from trading_system.core.logger import get_logger
from trading_system.risk.portfolio_risk import PortfolioState

logger = get_logger(__name__)


class RiskMonitor:
    """Monitors portfolio risk state in real time."""

    def check_emergency_conditions(self, state: PortfolioState) -> bool:
        """
        Returns True if emergency flatten should be triggered.
        Conditions:
          1. Drawdown >= MAX_DRAWDOWN_PCT * 1.5 (hard emergency)
          2. Consecutive losses >= CONSECUTIVE_LOSS_LIMIT * 2
        """
        from trading_system.config.settings import settings

        if state.peak_equity > 0:
            dd = (state.peak_equity - state.equity) / state.peak_equity
            if dd >= settings.MAX_DRAWDOWN_PCT * 1.5:
                logger.critical("EMERGENCY: drawdown critical",
                                extra={"drawdown_pct": dd})
                return True

        if state.consecutive_losses >= settings.CONSECUTIVE_LOSS_LIMIT * 2:
            logger.critical("EMERGENCY: consecutive losses critical",
                            extra={"losses": state.consecutive_losses})
            return True

        return False

    def update_daily_weekly_state(self, state: PortfolioState,
                                   current_equity: float) -> PortfolioState:
        """Update equity, PnL, and reset daily/weekly counters as needed."""
        from datetime import date

        today = date.today()
        if state.last_reset_date != today:
            state.daily_pnl = 0.0
            state.daily_start_equity = current_equity
            state.last_reset_date = today

        state.equity = current_equity
        if current_equity > state.peak_equity:
            state.peak_equity = current_equity

        state.daily_pnl = current_equity - state.daily_start_equity
        return state
