"""Module 1 infrastructure tests."""
import io
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure the repo root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSecretMasking:
    def test_api_key_masked_in_log_output(self):
        from trading_system.core.logger import get_logger, _mask_dict

        sensitive = {
            "api_key": "super_secret_key_12345",
            "message": "connecting to exchange",
        }
        masked = _mask_dict(sensitive)
        assert masked["api_key"] == "***REDACTED***"
        assert masked["message"] == "connecting to exchange"

    def test_api_secret_masked(self):
        from trading_system.core.logger import _mask_dict

        d = {"api_secret": "my_secret", "symbol": "BTCUSDT"}
        masked = _mask_dict(d)
        assert masked["api_secret"] == "***REDACTED***"
        assert masked["symbol"] == "BTCUSDT"

    def test_token_masked(self):
        from trading_system.core.logger import _mask_dict

        d = {"auth_token": "tok_abc123", "value": 42}
        masked = _mask_dict(d)
        assert masked["auth_token"] == "***REDACTED***"

    def test_nested_dict_masked(self):
        from trading_system.core.logger import _mask_dict

        d = {"credentials": {"api_key": "secret", "user": "alice"}}
        masked = _mask_dict(d)
        assert masked["credentials"]["api_key"] == "***REDACTED***"
        assert masked["credentials"]["user"] == "alice"


class TestExceptions:
    def test_base_exception_fields(self):
        from trading_system.core.exceptions import TradingSystemError

        err = TradingSystemError("test error", {"code": 42})
        assert err.message == "test error"
        assert err.context == {"code": 42}
        assert isinstance(err.timestamp, datetime)

    def test_exchange_error_hierarchy(self):
        from trading_system.core.exceptions import (
            ExchangeError, RateLimitError, AuthenticationError,
            InsufficientFundsError, OrderError, TradingSystemError,
        )

        for cls in (RateLimitError, AuthenticationError,
                    InsufficientFundsError, OrderError):
            err = cls("msg", {})
            assert isinstance(err, ExchangeError)
            assert isinstance(err, TradingSystemError)

    def test_risk_error_hierarchy(self):
        from trading_system.core.exceptions import (
            RiskError, DrawdownLimitError, DailyLossLimitError,
            PositionSizeError, CorrelationLimitError, TradingSystemError,
        )

        for cls in (DrawdownLimitError, DailyLossLimitError,
                    PositionSizeError, CorrelationLimitError):
            err = cls("msg", {})
            assert isinstance(err, RiskError)
            assert isinstance(err, TradingSystemError)

    def test_data_error_hierarchy(self):
        from trading_system.core.exceptions import (
            DataError, InsufficientDataError, StaleDataError,
        )

        assert issubclass(InsufficientDataError, DataError)
        assert issubclass(StaleDataError, DataError)

    def test_system_error_hierarchy(self):
        from trading_system.core.exceptions import (
            SystemError, HealthCheckError, LiveTradingGateError,
        )

        assert issubclass(HealthCheckError, SystemError)
        assert issubclass(LiveTradingGateError, SystemError)

    def test_exception_no_context_defaults_to_empty_dict(self):
        from trading_system.core.exceptions import TradingSystemError

        err = TradingSystemError("no context")
        assert err.context == {}


class TestSettings:
    def test_settings_load_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            from trading_system.config.settings import Settings
            s = Settings()
            assert s.BASE_RISK_PCT == 0.01
            assert s.MAX_CONCURRENT_TRADES == 6
            assert s.BINANCE_TESTNET is True

    def test_settings_override_from_env(self):
        with patch.dict(os.environ, {"BASE_RISK_PCT": "0.02",
                                      "MAX_CONCURRENT_TRADES": "4"}):
            from trading_system.config.settings import Settings
            s = Settings()
            assert s.BASE_RISK_PCT == 0.02
            assert s.MAX_CONCURRENT_TRADES == 4

    def test_live_trading_disabled_by_default(self):
        from trading_system.config.settings import Settings
        s = Settings()
        assert s.live_trading_active is False


class TestHalalCompliance:
    def test_direction_long_is_only_valid_direction(self):
        from trading_system.config.constants import Direction

        directions = list(Direction)
        assert len(directions) == 1
        assert Direction.LONG in directions

    def test_direction_long_value(self):
        from trading_system.config.constants import Direction

        assert Direction.LONG.value == "long"

    def test_trending_bear_allows_no_strategies(self):
        from trading_system.config.constants import (
            Regime, REGIME_ALLOWED_STRATEGIES,
        )

        assert REGIME_ALLOWED_STRATEGIES[Regime.TRENDING_BEAR] == []

    def test_high_vol_chaos_allows_no_strategies(self):
        from trading_system.config.constants import (
            Regime, REGIME_ALLOWED_STRATEGIES,
        )

        assert REGIME_ALLOWED_STRATEGIES[Regime.HIGH_VOL_CHAOS] == []

    def test_bear_risk_multiplier_is_zero(self):
        from trading_system.config.constants import (
            Regime, REGIME_RISK_MULTIPLIERS,
        )

        assert REGIME_RISK_MULTIPLIERS[Regime.TRENDING_BEAR] == 0.0
        assert REGIME_RISK_MULTIPLIERS[Regime.HIGH_VOL_CHAOS] == 0.0
