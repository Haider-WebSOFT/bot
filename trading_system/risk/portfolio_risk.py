"""Portfolio-level risk checks for LONG-only positions."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date

from trading_system.config.settings import settings
from trading_system.core.exceptions import (
    DailyLossLimitError, DrawdownLimitError, CorrelationLimitError,
)
from trading_system.core.logger import get_logger
from trading_system.risk.position_sizer import SizingResult

logger = get_logger(__name__)


@dataclass
class PortfolioState:
    """Current portfolio state snapshot."""
    equity: float
    peak_equity: float
    open_positions: dict[str, dict]   # symbol -> position info
    daily_pnl: float = 0.0
    daily_start_equity: float = 0.0
    weekly_pnl: float = 0.0
    weekly_start_equity: float = 0.0
    consecutive_losses: int = 0
    trading_halted: bool = False
    halt_reason: str = ""
    last_reset_date: date = field(default_factory=date.today)


class PortfolioRiskEngine:
    """Checks portfolio-level risk limits before allowing new LONG trades."""

    def can_open_trade(
        self,
        state: PortfolioState,
        new_sizing: SizingResult,
        new_symbol: str,
    ) -> tuple[bool, str]:
        """Return (allowed, reason). All limits checked in order."""

        if state.trading_halted:
            return False, f"Trading halted: {state.halt_reason}"

        # 1. Daily loss limit
        if state.daily_start_equity > 0:
            daily_loss_pct = -state.daily_pnl / state.daily_start_equity
            if daily_loss_pct >= settings.DAILY_LOSS_LIMIT_PCT:
                state.trading_halted = True
                state.halt_reason = f"Daily loss {daily_loss_pct:.1%} >= limit"
                logger.warning("Daily loss limit hit", extra={"pct": daily_loss_pct})
                return False, state.halt_reason

        # 2. Max drawdown
        if state.peak_equity > 0:
            dd = (state.peak_equity - state.equity) / state.peak_equity
            if dd >= settings.MAX_DRAWDOWN_PCT:
                state.trading_halted = True
                state.halt_reason = f"Max drawdown {dd:.1%} >= {settings.MAX_DRAWDOWN_PCT:.1%}"
                logger.critical("Max drawdown halt", extra={"drawdown_pct": dd})
                return False, state.halt_reason

        # 3. Max concurrent trades
        if len(state.open_positions) >= settings.MAX_CONCURRENT_TRADES:
            return False, f"Max concurrent trades ({settings.MAX_CONCURRENT_TRADES}) reached"

        # 4. Portfolio exposure
        total_exposure = sum(
            p.get("value", 0) for p in state.open_positions.values()
        )
        new_value = new_sizing.quantity * new_sizing.entry_price
        if (total_exposure + new_value) / max(state.equity, 1e-9) > settings.MAX_PORTFOLIO_EXPOSURE:
            return False, "Portfolio exposure would exceed 25%"

        # 5. Correlation (simplified: count same-asset-class positions)
        correlated_count = sum(
            1 for sym in state.open_positions
            if sym.endswith("USDT") and new_symbol.endswith("USDT")
        )
        if correlated_count >= settings.MAX_CORRELATED_POSITIONS:
            return False, f"Correlation limit: {correlated_count} USDT pairs open"

        # 6. Weekly loss (allow but note for sizing)
        if state.weekly_start_equity > 0:
            weekly_loss_pct = -state.weekly_pnl / state.weekly_start_equity
            if weekly_loss_pct >= settings.WEEKLY_LOSS_LIMIT_PCT:
                logger.warning("Weekly loss — half sizing applied",
                               extra={"pct": weekly_loss_pct})

        logger.info("Trade approved by portfolio risk",
                    extra={"symbol": new_symbol,
                           "open_positions": len(state.open_positions)})
        return True, "approved"
