"""Module 2 client tests."""
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBinanceClientSideguard:
    def test_sell_order_raises_value_error_when_side_invalid(self):
        """BinanceClient must reject invalid side strings."""
        with patch("binance.client.Client") as MockClient:
            MockClient.return_value = MagicMock()
            from trading_system.clients.binance_client import BinanceClient
            client = BinanceClient()

            async def _run():
                with pytest.raises(ValueError, match="Invalid side"):
                    await client.place_limit_order("BTCUSDT", "INVALID", 0.001, 50000.0)

            import asyncio
            asyncio.get_event_loop().run_until_complete(_run())


class TestAlpacaClientOHLCV:
    def test_get_ohlcv_returns_correct_columns(self):
        """AlpacaClient.get_ohlcv returns DataFrame with required columns."""
        import yfinance as yf
        mock_df = pd.DataFrame({
            "Datetime": pd.date_range("2024-01-01", periods=10, freq="15min"),
            "Open": [100.0] * 10,
            "High": [101.0] * 10,
            "Low": [99.0] * 10,
            "Close": [100.5] * 10,
            "Volume": [1000] * 10,
        })

        with patch("alpaca_trade_api.REST") as MockREST:
            MockREST.return_value = MagicMock()
            with patch("yfinance.Ticker") as MockTicker:
                MockTicker.return_value.history.return_value = mock_df.set_index("Datetime")
                from trading_system.clients.alpaca_client import AlpacaClient
                client = AlpacaClient()

                import asyncio
                df = asyncio.get_event_loop().run_until_complete(
                    client.get_ohlcv("AAPL", "15m", limit=10)
                )

        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        assert required_cols.issubset(set(df.columns))
        assert len(df) <= 10


class TestHealthReport:
    def test_health_report_dataclass_creation(self):
        from trading_system.clients.health import HealthReport

        report = HealthReport(
            status="healthy",
            binance_ok=True,
            alpaca_ok=True,
            database_ok=True,
            latency_ms={"binance_ms": 45.0, "alpaca_ms": 80.0},
            errors=[],
            timestamp=datetime.utcnow(),
        )
        assert report.status == "healthy"
        assert report.binance_ok is True
        assert report.latency_ms["binance_ms"] == 45.0

    def test_health_report_critical_when_exchange_fails(self):
        from trading_system.clients.health import HealthReport

        report = HealthReport(
            status="critical",
            binance_ok=False,
            alpaca_ok=True,
            database_ok=True,
            latency_ms={},
            errors=["Binance: connection refused"],
            timestamp=datetime.utcnow(),
        )
        assert report.status == "critical"
        assert len(report.errors) > 0


class TestHealthCheckerMocked:
    def test_check_all_measures_latency(self):
        """HealthChecker records latency for each component."""
        with patch("binance.client.Client") as MockBinance:
            MockBinance.return_value = MagicMock()
            with patch("alpaca_trade_api.REST") as MockAlpaca:
                MockAlpaca.return_value = MagicMock()

                from trading_system.clients.binance_client import BinanceClient
                from trading_system.clients.alpaca_client import AlpacaClient
                from trading_system.clients.health import HealthChecker

                binance = BinanceClient()
                alpaca = AlpacaClient()

                binance.get_account_balance = AsyncMock(return_value={"USDT": 10.0})
                alpaca.get_account = AsyncMock(return_value={"equity": 10.0})

                checker = HealthChecker()
                import asyncio
                report = asyncio.get_event_loop().run_until_complete(
                    checker.check_all(binance, alpaca)
                )

        assert "binance_ms" in report.latency_ms
        assert "alpaca_ms" in report.latency_ms
        assert report.latency_ms["binance_ms"] >= 0
