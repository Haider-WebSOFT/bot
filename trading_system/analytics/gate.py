"""Live trading gate — all criteria must pass before live trading is allowed."""
from __future__ import annotations

from dataclasses import dataclass

from trading_system.backtest.metrics import MetricsResult
from trading_system.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GateResult:
    """Result of the live trading gate check."""
    approved: bool
    criteria_met: dict[str, bool]
    criteria_values: dict[str, float]
    criteria_required: dict[str, float]
    blockers: list[str]
    message: str


class LiveTradingGate:
    """
    Non-negotiable hard gate before any live trading.
    ALL criteria must be True for approved=True.
    Cannot be bypassed by config flags alone.
    """

    REQUIREMENTS: dict[str, float] = {
        "min_trades":       50.0,
        "profit_factor":    1.30,
        "sharpe_ratio":     1.00,
        "max_drawdown_pct": 0.15,
        "min_win_rate":     0.40,
        "min_regimes":      2.0,
        "min_expectancy":   0.003,
    }

    def check(self, metrics: MetricsResult, trades: list[dict]) -> GateResult:
        """Evaluate all gate criteria. Returns GateResult."""
        hold_cash = {"trending_bear", "high_vol_chaos"}
        regimes_traded = len(set(
            t.get("regime", "unknown") for t in trades
            if t.get("regime") not in hold_cash
            and t.get("exit_price") is not None
        ))

        values: dict[str, float] = {
            "min_trades":       float(metrics.total_trades),
            "profit_factor":    metrics.profit_factor,
            "sharpe_ratio":     metrics.sharpe_ratio,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "min_win_rate":     metrics.win_rate,
            "min_regimes":      float(regimes_traded),
            "min_expectancy":   metrics.expectancy,
        }

        met: dict[str, bool] = {
            "min_trades":
                values["min_trades"] >= self.REQUIREMENTS["min_trades"],
            "profit_factor":
                values["profit_factor"] >= self.REQUIREMENTS["profit_factor"],
            "sharpe_ratio":
                values["sharpe_ratio"] >= self.REQUIREMENTS["sharpe_ratio"],
            "max_drawdown_pct":
                values["max_drawdown_pct"] <= self.REQUIREMENTS["max_drawdown_pct"],
            "min_win_rate":
                values["min_win_rate"] >= self.REQUIREMENTS["min_win_rate"],
            "min_regimes":
                values["min_regimes"] >= self.REQUIREMENTS["min_regimes"],
            "min_expectancy":
                values["min_expectancy"] >= self.REQUIREMENTS["min_expectancy"],
        }

        blockers = [k for k, v in met.items() if not v]
        approved = len(blockers) == 0

        if approved:
            msg = "✓ All gate criteria met — live trading ALLOWED (requires manual flag)"
        else:
            msg = (
                f"✗ NO VALID EDGE FOUND — LIVE TRADING BLOCKED. "
                f"Blockers: {', '.join(blockers)}"
            )

        logger.info("Gate check complete",
                    extra={"approved": approved, "blockers": blockers})
        return GateResult(
            approved=approved,
            criteria_met=met,
            criteria_values=values,
            criteria_required=self.REQUIREMENTS,
            blockers=blockers,
            message=msg,
        )
