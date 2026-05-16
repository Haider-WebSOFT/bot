"""Walk-forward validation to prevent overfitting."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from trading_system.backtest.metrics import MetricsResult, compute_metrics
from trading_system.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WalkForwardResult:
    """Results from a walk-forward validation run."""
    n_windows: int
    train_metrics: list[MetricsResult]
    test_metrics: list[MetricsResult]
    combined_test_metrics: MetricsResult
    degradation_pct: float   # How much worse test is vs train


class WalkForwardValidator:
    """Splits data into train/test windows and validates out-of-sample."""

    def run(
        self,
        ohlcv_data: dict,
        start_date: datetime,
        end_date: datetime,
        train_pct: float = 0.70,
        n_windows: int = 3,
        initial_capital: float = 10.0,
    ) -> WalkForwardResult:
        from trading_system.backtest.engine import BacktestEngine

        total_days = (end_date - start_date).days
        window_days = total_days // n_windows
        step_days = window_days

        train_results: list[MetricsResult] = []
        test_results: list[MetricsResult] = []
        all_test_trades: list[dict] = []
        all_test_equity: list[float] = [initial_capital]

        engine = BacktestEngine()

        for w in range(n_windows):
            win_start = start_date + timedelta(days=w * step_days)
            win_end = win_start + timedelta(days=window_days)
            train_end = win_start + timedelta(days=int(window_days * train_pct))

            logger.info("Walk-forward window",
                        extra={"window": w, "train_end": str(train_end),
                               "test_end": str(win_end)})

            try:
                train_bt = engine.run(ohlcv_data, win_start, train_end,
                                      initial_capital=initial_capital)
                train_results.append(train_bt.metrics)

                test_bt = engine.run(ohlcv_data, train_end, win_end,
                                     initial_capital=initial_capital)
                test_results.append(test_bt.metrics)
                all_test_trades.extend(test_bt.trades)
                all_test_equity.extend(test_bt.equity_curve[1:])
            except Exception as exc:
                logger.error("Walk-forward window failed",
                             extra={"window": w, "error": str(exc)})

        combined = compute_metrics(all_test_trades, all_test_equity, initial_capital)

        # Degradation: how much sharpe dropped from train to test
        avg_train_sharpe = sum(m.sharpe_ratio for m in train_results) / max(len(train_results), 1)
        degradation = ((avg_train_sharpe - combined.sharpe_ratio) / max(abs(avg_train_sharpe), 1e-9))

        if degradation > 0.5:
            logger.warning("Walk-forward degradation > 50%",
                           extra={"degradation_pct": degradation * 100})

        return WalkForwardResult(
            n_windows=n_windows,
            train_metrics=train_results,
            test_metrics=test_results,
            combined_test_metrics=combined,
            degradation_pct=degradation * 100,
        )
