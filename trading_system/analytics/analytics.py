"""Live analytics from paper trading journal."""
from __future__ import annotations

from trading_system.backtest.metrics import MetricsResult, compute_metrics
from trading_system.core.logger import get_logger

logger = get_logger(__name__)


class LiveAnalytics:
    """Computes live performance metrics from the trade journal."""

    def compute_live_metrics(self, journal) -> MetricsResult:
        """Full MetricsResult from journal trades and equity snapshots."""
        trades = journal.get_all_trades()
        closed = [t for t in trades if t.get("exit_price") is not None]

        equity_curve = [10.0]
        try:
            from trading_system.paper_trading.trade_journal import EquitySnapshot
            from sqlalchemy.orm import Session
            with Session(journal._engine) as sess:
                snaps = sess.query(EquitySnapshot).order_by(EquitySnapshot.id).all()
                if snaps:
                    equity_curve = [s.equity for s in snaps]
        except Exception:
            pass

        return compute_metrics(closed, equity_curve)

    def get_regime_performance(self, journal) -> dict:
        """Win rate, expectancy, PF split by regime."""
        trades = [t for t in journal.get_all_trades() if t.get("exit_price")]
        result: dict[str, dict] = {}
        for regime in set(t.get("regime", "unknown") for t in trades):
            subset = [t for t in trades if t.get("regime") == regime]
            wins = [t for t in subset if (t.get("pnl_pct") or 0) > 0]
            losses = [t for t in subset if (t.get("pnl_pct") or 0) <= 0]
            win_rate = len(wins) / len(subset) if subset else 0.0
            avg_win = sum(t.get("pnl_pct", 0) for t in wins) / max(len(wins), 1)
            avg_loss = sum(t.get("pnl_pct", 0) for t in losses) / max(len(losses), 1)
            result[regime] = {
                "trades": len(subset),
                "win_rate": win_rate,
                "expectancy": win_rate * avg_win + (1 - win_rate) * avg_loss,
            }
        return result
