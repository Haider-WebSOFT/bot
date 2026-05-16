"""Custom exception hierarchy for the trading system."""
from datetime import datetime


class TradingSystemError(Exception):
    """Base exception for all trading system errors."""

    def __init__(self, message: str, context: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.timestamp = datetime.utcnow()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, context={self.context!r})"


# Exchange errors
class ExchangeError(TradingSystemError):
    """Base class for exchange-related errors."""


class RateLimitError(ExchangeError):
    """Exchange rate limit exceeded."""


class AuthenticationError(ExchangeError):
    """Invalid API credentials."""


class InsufficientFundsError(ExchangeError):
    """Insufficient balance for the requested order."""


class OrderError(ExchangeError):
    """Generic order placement or management error."""


# Risk errors
class RiskError(TradingSystemError):
    """Base class for risk management errors."""


class DrawdownLimitError(RiskError):
    """Maximum drawdown limit reached."""


class DailyLossLimitError(RiskError):
    """Daily loss limit reached."""


class PositionSizeError(RiskError):
    """Position size calculation error or constraint violation."""


class CorrelationLimitError(RiskError):
    """Correlation limit would be exceeded by new position."""


# Data errors
class DataError(TradingSystemError):
    """Base class for data-related errors."""


class InsufficientDataError(DataError):
    """Not enough historical bars to compute indicators."""


class StaleDataError(DataError):
    """Data is too old to be used for trading decisions."""


# System errors
class SystemError(TradingSystemError):
    """Base class for system-level errors."""


class HealthCheckError(SystemError):
    """Critical system health check failed."""


class LiveTradingGateError(SystemError):
    """Live trading gate criteria not met."""
