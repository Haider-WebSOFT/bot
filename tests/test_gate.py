"""Module 14 live trading gate tests."""
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_metrics(**kwargs):
    from trading_system.backtest.metrics import MetricsResult

    defaults = dict(
        total_trades=60, win_rate=0.48, avg_win_pct=0.025, avg_loss_pct=-0.015,
        avg_rr=1.67, expectancy=0.005, profit_factor=1.6,
        total_return_pct=0.15, cagr_pct=0.30, sharpe_ratio=1.2,
        sortino_ratio=1.5, calmar_ratio=2.0, max_drawdown_pct=0.10,
        max_drawdown_duration_bars=100, avg_trade_duration_bars=20.0,
        regime_breakdown={}, monthly_returns={},
        mae_avg=-0.01, mfe_avg=0.025, mae_mfe_ratio=0.4,
        strategy_breakdown={
            "standard": {"trades": 50, "win_rate": 0.48, "expectancy": 0.005},
            "early_momentum": {"trades": 10, "win_rate": 0.50, "expectancy": 0.006},
        },
    )
    defaults.update(kwargs)
    return MetricsResult(**defaults)


def _make_trades(n: int = 60, regimes: list[str] | None = None) -> list[dict]:
    if regimes is None:
        regimes = ["trending_bull", "ranging"]
    return [{"regime": regimes[i % len(regimes)], "exit_price": 101.0, "pnl_pct": 0.01}
            for i in range(n)]


class TestLiveTradingGate:
    def test_all_criteria_met_approved(self):
        from trading_system.analytics.gate import LiveTradingGate

        gate = LiveTradingGate()
        result = gate.check(_make_metrics(), _make_trades(60, ["trending_bull", "ranging"]))
        assert result.approved is True
        assert result.blockers == []

    def test_low_sharpe_blocks(self):
        from trading_system.analytics.gate import LiveTradingGate

        gate = LiveTradingGate()
        result = gate.check(_make_metrics(sharpe_ratio=0.8), _make_trades())
        assert result.approved is False
        assert "sharpe_ratio" in result.blockers

    def test_insufficient_trades_blocks(self):
        from trading_system.analytics.gate import LiveTradingGate

        gate = LiveTradingGate()
        result = gate.check(_make_metrics(total_trades=30), _make_trades(30))
        assert result.approved is False
        assert "min_trades" in result.blockers

    def test_two_distinct_regimes_passes(self):
        from trading_system.analytics.gate import LiveTradingGate

        gate = LiveTradingGate()
        result = gate.check(_make_metrics(),
                            _make_trades(60, ["trending_bull", "ranging"]))
        assert result.criteria_met["min_regimes"] is True

    def test_trending_bear_not_counted_in_regimes(self):
        from trading_system.analytics.gate import LiveTradingGate

        gate = LiveTradingGate()
        result = gate.check(_make_metrics(),
                            _make_trades(60, ["trending_bear", "high_vol_chaos"]))
        assert result.criteria_values["min_regimes"] == 0.0
        assert result.criteria_met["min_regimes"] is False
