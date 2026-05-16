"""ATR-based position sizing for LONG-only halal trades."""
from __future__ import annotations

from dataclasses import dataclass

from trading_system.agents.regime_detector import RegimeResult
from trading_system.config.constants import Direction
from trading_system.config.settings import settings
from trading_system.core.exceptions import PositionSizeError
from trading_system.core.logger import get_logger
from trading_system.data.models import MarketSnapshot

logger = get_logger(__name__)


@dataclass
class SizingResult:
    """Output of the position sizer for a single LONG trade."""
    symbol: str
    direction: Direction      # Always LONG
    quantity: float
    entry_price: float
    stop_loss: float          # Always < entry_price
    take_profit: float        # Always > entry_price
    risk_dollars: float
    r_r_ratio: float
    atr_used: float
    regime_multiplier: float
    size_reduction_reason: str | None


class PositionSizer:
    """Calculates ATR-based position sizes for LONG entries."""

    def __init__(self, consecutive_losses: int = 0,
                 weekly_loss_pct: float = 0.0) -> None:
        self.consecutive_losses = consecutive_losses
        self.weekly_loss_pct = weekly_loss_pct

    def calculate(
        self,
        snap: MarketSnapshot,
        regime: RegimeResult,
        account_equity: float,
        strategy_type: str = "standard",
    ) -> SizingResult:
        """Calculate position size for a LONG trade."""
        entry_price = snap.current_price
        atr = snap.tf_1h.atr

        sl_mult = settings.ATR_SL_MULTIPLIER
        tp_mult = settings.ATR_TP_MULTIPLIER

        if strategy_type == "early_momentum":
            sl_mult = sl_mult * 0.8  # tighter stop

        sl_distance = atr * sl_mult
        tp_distance = atr * tp_mult

        stop_loss = entry_price - sl_distance
        take_profit = entry_price + tp_distance

        # Assertions: LONG only
        if stop_loss >= entry_price:
            raise ValueError(
                f"stop_loss {stop_loss:.4f} must be < entry_price {entry_price:.4f}"
            )
        if take_profit <= entry_price:
            raise ValueError(
                f"take_profit {take_profit:.4f} must be > entry_price {entry_price:.4f}"
            )

        risk_dollars = account_equity * settings.BASE_RISK_PCT * regime.risk_multiplier
        quantity = risk_dollars / max(sl_distance, 1e-9)

        reduction_reasons: list[str] = []

        # Volatility adjustment
        atr_pct = snap.tf_1h.atr_pct
        if atr_pct > 3.0:
            quantity *= 0.70
            reduction_reasons.append("ATR>3% volatility -30%")

        # Regime multiplier already applied to risk_dollars above

        # Consecutive losses
        if self.consecutive_losses >= settings.CONSECUTIVE_LOSS_LIMIT:
            quantity *= 0.50
            reduction_reasons.append(f"consecutive_losses={self.consecutive_losses} -50%")

        # Weekly loss
        if self.weekly_loss_pct > settings.WEEKLY_LOSS_LIMIT_PCT:
            quantity *= 0.50
            reduction_reasons.append(f"weekly_loss={self.weekly_loss_pct:.1%} -50%")

        # Hard cap: max position size
        max_position_value = account_equity * settings.MAX_POSITION_PCT
        if quantity * entry_price > max_position_value:
            quantity = max_position_value / max(entry_price, 1e-9)
            reduction_reasons.append("max_position_pct cap applied")

        if risk_dollars < 0.10:
            logger.warning("Small account sizing",
                           extra={"risk_dollars": risk_dollars,
                                  "account_equity": account_equity})

        if quantity <= 0:
            raise PositionSizeError(
                "Quantity <= 0 after adjustments",
                {"quantity": quantity, "account_equity": account_equity},
            )

        rr = tp_distance / max(sl_distance, 1e-9)

        return SizingResult(
            symbol=snap.symbol,
            direction=Direction.LONG,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_dollars=risk_dollars,
            r_r_ratio=rr,
            atr_used=atr,
            regime_multiplier=regime.risk_multiplier,
            size_reduction_reason="; ".join(reduction_reasons) if reduction_reasons else None,
        )
