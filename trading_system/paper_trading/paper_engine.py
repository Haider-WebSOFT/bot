"""Paper trading simulation engine."""
from __future__ import annotations

from datetime import datetime

from trading_system.config.settings import settings
from trading_system.core.logger import get_logger
from trading_system.scanner.ranker import OpportunityCard

logger = get_logger(__name__)


class PaperTradingEngine:
    """Simulates LONG trade execution with realistic slippage."""

    def __init__(
        self,
        initial_capital: float,
        journal,
        portfolio_risk,
        trade_manager,
        execution_agent,
    ) -> None:
        self.equity = initial_capital
        self.initial_capital = initial_capital
        self._journal = journal
        self._portfolio_risk = portfolio_risk
        self._trade_manager = trade_manager
        self._execution_agent = execution_agent
        self._open_positions: dict[int, dict] = {}
        self._peak_equity = initial_capital

    async def process_opportunity(
        self, card: OpportunityCard, snap, sizing
    ) -> bool:
        """Simulate LONG entry with realistic slippage."""
        base_slip = settings.SLIPPAGE_PCT
        if snap.tf_15m.rvol > 2.0:
            base_slip *= 1.5
        if snap.spread_pct > 0.001:
            base_slip += snap.spread_pct / 2

        actual_entry = card.suggested_entry * (1 + base_slip)

        from trading_system.risk.portfolio_risk import PortfolioState
        state = PortfolioState(
            equity=self.equity,
            peak_equity=self._peak_equity,
            open_positions={str(k): {"value": v.get("value", 0)}
                            for k, v in self._open_positions.items()},
            daily_start_equity=self.initial_capital,
            weekly_start_equity=self.initial_capital,
        )
        allowed, reason = self._portfolio_risk.can_open_trade(state, sizing, card.symbol)
        if not allowed:
            logger.info("Paper trade rejected", extra={"reason": reason})
            return False

        trade_id = self._journal.open_trade(card, sizing, actual_entry)
        self._open_positions[trade_id] = {
            "symbol": card.symbol,
            "entry_price": actual_entry,
            "stop_loss": sizing.stop_loss,
            "take_profit": sizing.take_profit,
            "quantity": sizing.quantity,
            "entry_time": datetime.utcnow(),
            "strategy_type": card.strategy_type,
            "value": actual_entry * sizing.quantity,
            "mae": 0.0,
            "mfe": 0.0,
        }
        self.equity -= actual_entry * sizing.quantity * settings.COMMISSION_PCT
        logger.info("Paper trade opened",
                    extra={"trade_id": trade_id, "symbol": card.symbol,
                           "entry": actual_entry, "qty": sizing.quantity})
        return True

    async def update_all_positions(self, snapshots: list) -> None:
        """Update all open LONG positions on each bar close."""
        snap_map = {s.symbol: s for s in snapshots}
        closed_ids: list[int] = []

        for trade_id, pos in list(self._open_positions.items()):
            snap = snap_map.get(pos["symbol"])
            if not snap:
                continue

            updated = self._trade_manager.update_position(pos, snap)
            self._open_positions[trade_id] = updated

            if "exit_signal" in updated:
                slip = settings.SLIPPAGE_PCT
                actual_exit = snap.current_price * (1 - slip)
                pnl = (actual_exit - updated["entry_price"]) * updated["quantity"]
                commission = actual_exit * updated["quantity"] * settings.COMMISSION_PCT
                self.equity += pnl - commission
                self._peak_equity = max(self._peak_equity, self.equity)

                self._journal.close_trade(
                    trade_id, actual_exit,
                    updated["exit_signal"].value,
                    updated.get("mae", 0.0),
                    updated.get("mfe", 0.0),
                )
                self._execution_agent.record_execution(
                    updated["entry_price"], actual_exit, True, 50.0
                )
                closed_ids.append(trade_id)

        for tid in closed_ids:
            del self._open_positions[tid]

        dd = ((self._peak_equity - self.equity) / self._peak_equity
              if self._peak_equity > 0 else 0.0)
        self._journal.update_equity_snapshot(self.equity, dd)
