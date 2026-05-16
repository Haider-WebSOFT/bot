"""Parameter optimizer with train/test split guard against overfitting."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from trading_system.backtest.metrics import MetricsResult
from trading_system.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """Best parameters found with train and test metrics."""
    best_params: dict[str, Any]
    train_metrics: MetricsResult
    test_metrics: MetricsResult
    overfitting_warning: bool


class StrategyOptimizer:
    """Grid-search optimizer with mandatory out-of-sample validation."""

    def optimize(
        self,
        ohlcv_data: dict,
        param_grid: dict[str, list],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 10.0,
        train_pct: float = 0.70,
    ) -> OptimizationResult:
        from trading_system.backtest.engine import BacktestEngine

        total_days = (end_date - start_date).days
        train_end = start_date + __import__("datetime").timedelta(
            days=int(total_days * train_pct)
        )

        engine = BacktestEngine()
        best_sharpe = float("-inf")
        best_params: dict = {}
        best_train: MetricsResult | None = None

        import itertools
        keys = list(param_grid.keys())
        for combo in itertools.product(*param_grid.values()):
            params = dict(zip(keys, combo))
            try:
                result = engine.run(ohlcv_data, start_date, train_end,
                                    initial_capital=initial_capital)
                if result.metrics.sharpe_ratio > best_sharpe:
                    best_sharpe = result.metrics.sharpe_ratio
                    best_params = params
                    best_train = result.metrics
            except Exception as exc:
                logger.debug("Optimization combo failed",
                             extra={"params": params, "error": str(exc)})

        if best_train is None:
            from trading_system.backtest.metrics import _empty_metrics
            best_train = _empty_metrics()

        # Out-of-sample test
        test_result = engine.run(ohlcv_data, train_end, end_date,
                                 initial_capital=initial_capital)
        test_metrics = test_result.metrics

        degradation = best_sharpe - test_metrics.sharpe_ratio
        overfitting_warning = degradation > best_sharpe * 0.5

        if overfitting_warning:
            logger.warning("Possible overfitting or insufficient sample size",
                           extra={"train_sharpe": best_sharpe,
                                  "test_sharpe": test_metrics.sharpe_ratio})

        return OptimizationResult(
            best_params=best_params,
            train_metrics=best_train,
            test_metrics=test_metrics,
            overfitting_warning=overfitting_warning,
        )
