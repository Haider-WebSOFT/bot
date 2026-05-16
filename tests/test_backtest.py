"""Module 11 backtest tests."""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_bull_ohlcv(n: int = 400, base: float = 100.0, trend: float = 0.001) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    prices = base * np.cumprod(1 + trend + rng.normal(0, 0.005, n))
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame({
        "timestamp": ts,
        "open": prices * 0.999,
        "high": prices * 1.003,
        "low": prices * 0.997,
        "close": prices,
        "volume": rng.uniform(1000, 5000, n),
    })


def _make_bear_ohlcv(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    prices = 100.0 * np.cumprod(1 - 0.001 + rng.normal(0, 0.005, n))
    prices = np.maximum(prices, 0.01)
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame({
        "timestamp": ts,
        "open": prices * 1.001,
        "high": prices * 1.003,
        "low": prices * 0.997,
        "close": prices,
        "volume": rng.uniform(500, 2000, n),
    })


def _make_chaos_ohlcv(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.05, n))
    prices = np.maximum(prices, 1.0)
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame({
        "timestamp": ts,
        "open": prices * 0.98,
        "high": prices * 1.08,
        "low": prices * 0.92,
        "close": prices,
        "volume": rng.uniform(500, 3000, n),
    })


def _make_ohlcv_set(df_15m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = df_15m.set_index("timestamp")
    df_1h = df.resample("1h").agg({"open": "first", "high": "max",
                                    "low": "min", "close": "last",
                                    "volume": "sum"}).dropna()
    df_4h = df.resample("4h").agg({"open": "first", "high": "max",
                                    "low": "min", "close": "last",
                                    "volume": "sum"}).dropna()
    return {"15m": df, "1h": df_1h, "4h": df_4h}


class TestBacktestEngine:
    def test_bear_period_opens_zero_trades(self):
        from trading_system.backtest.engine import BacktestEngine

        df = _make_bear_ohlcv(400)
        data = {"BTCUSDT": _make_ohlcv_set(df)}
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 4)

        engine = BacktestEngine()
        result = engine.run(data, start, end, initial_capital=10.0)
        assert result.metrics.total_trades == 0

    def test_chaos_period_opens_zero_trades(self):
        from trading_system.backtest.engine import BacktestEngine

        df = _make_chaos_ohlcv(400)
        data = {"BTCUSDT": _make_ohlcv_set(df)}
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 4)

        engine = BacktestEngine()
        result = engine.run(data, start, end, initial_capital=10.0)
        assert result.metrics.total_trades == 0

    def test_equity_curve_is_populated(self):
        from trading_system.backtest.engine import BacktestEngine

        df = _make_bull_ohlcv(300)
        data = {"BTCUSDT": _make_ohlcv_set(df)}
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3, 3, 45)

        engine = BacktestEngine()
        result = engine.run(data, start, end, initial_capital=10.0)
        assert len(result.equity_curve) > 0
        assert result.final_equity > 0


class TestMetrics:
    def test_strategy_breakdown_has_both_keys(self):
        from trading_system.backtest.metrics import compute_metrics

        trades = [
            {"pnl_pct": 0.05, "pnl_dollar": 0.5, "strategy_type": "standard",
             "regime": "trending_bull", "mae": -0.01, "mfe": 0.06, "duration_bars": 10},
            {"pnl_pct": 0.03, "pnl_dollar": 0.3, "strategy_type": "early_momentum",
             "regime": "trending_bull", "mae": -0.005, "mfe": 0.04, "duration_bars": 5},
        ]
        equity = [10.0, 10.5, 10.8]
        metrics = compute_metrics(trades, equity)
        assert "early_momentum" in metrics.strategy_breakdown
        assert "standard" in metrics.strategy_breakdown

    def test_high_profit_factor_logs_warning(self, caplog):
        import logging
        from trading_system.backtest.metrics import compute_metrics

        trades = [
            {"pnl_pct": 0.50, "pnl_dollar": 5.0, "strategy_type": "standard",
             "regime": "trending_bull", "mae": 0.0, "mfe": 0.5, "duration_bars": 1}
            for _ in range(5)
        ] + [
            {"pnl_pct": -0.01, "pnl_dollar": -0.1, "strategy_type": "standard",
             "regime": "trending_bull", "mae": -0.01, "mfe": 0.0, "duration_bars": 1}
        ]
        equity = [10.0 + i for i in range(len(trades) + 1)]

        with caplog.at_level(logging.WARNING):
            metrics = compute_metrics(trades, equity)

        assert metrics.profit_factor > 3.0
        assert any("overfitting" in r.message.lower() for r in caplog.records)
