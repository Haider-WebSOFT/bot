"""Trading system configuration loaded from environment variables."""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

def _bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")

def _float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))

def _int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))

def _str(key: str, default: str = "") -> str:
    return os.getenv(key, default)


@dataclass
class Settings:
    # Exchange credentials
    BINANCE_API_KEY: str = field(default_factory=lambda: _str("BINANCE_API_KEY"))
    BINANCE_API_SECRET: str = field(default_factory=lambda: _str("BINANCE_API_SECRET"))
    BINANCE_TESTNET: bool = field(default_factory=lambda: _bool("BINANCE_TESTNET", True))
    BINANCE_TESTNET_URL: str = "https://testnet.binance.vision"

    ALPACA_API_KEY: str = field(default_factory=lambda: _str("ALPACA_API_KEY"))
    ALPACA_API_SECRET: str = field(default_factory=lambda: _str("ALPACA_API_SECRET"))
    ALPACA_BASE_URL: str = field(default_factory=lambda: _str("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"))

    # Trading parameters
    BASE_RISK_PCT: float = field(default_factory=lambda: _float("BASE_RISK_PCT", 0.01))
    MAX_POSITION_PCT: float = field(default_factory=lambda: _float("MAX_POSITION_PCT", 0.05))
    MAX_PORTFOLIO_EXPOSURE: float = field(default_factory=lambda: _float("MAX_PORTFOLIO_EXPOSURE", 0.25))
    MAX_CONCURRENT_TRADES: int = field(default_factory=lambda: _int("MAX_CONCURRENT_TRADES", 6))
    MAX_CORRELATED_POSITIONS: int = field(default_factory=lambda: _int("MAX_CORRELATED_POSITIONS", 2))
    CORRELATION_THRESHOLD: float = field(default_factory=lambda: _float("CORRELATION_THRESHOLD", 0.70))

    # Risk limits
    DAILY_LOSS_LIMIT_PCT: float = field(default_factory=lambda: _float("DAILY_LOSS_LIMIT_PCT", 0.03))
    WEEKLY_LOSS_LIMIT_PCT: float = field(default_factory=lambda: _float("WEEKLY_LOSS_LIMIT_PCT", 0.07))
    MAX_DRAWDOWN_PCT: float = field(default_factory=lambda: _float("MAX_DRAWDOWN_PCT", 0.15))
    CONSECUTIVE_LOSS_LIMIT: int = field(default_factory=lambda: _int("CONSECUTIVE_LOSS_LIMIT", 4))

    # ATR multipliers
    ATR_SL_MULTIPLIER: float = field(default_factory=lambda: _float("ATR_SL_MULTIPLIER", 1.5))
    ATR_TP_MULTIPLIER: float = field(default_factory=lambda: _float("ATR_TP_MULTIPLIER", 3.0))
    ATR_TRAILING_MULTIPLIER: float = field(default_factory=lambda: _float("ATR_TRAILING_MULTIPLIER", 0.5))
    ATR_BREAKEVEN_TRIGGER: float = field(default_factory=lambda: _float("ATR_BREAKEVEN_TRIGGER", 1.0))

    # Scoring thresholds
    ENTRY_SCORE_MIN: float = field(default_factory=lambda: _float("ENTRY_SCORE_MIN", 60.0))
    ENTRY_CONFIDENCE_MIN: float = field(default_factory=lambda: _float("ENTRY_CONFIDENCE_MIN", 0.65))
    STRONG_ENTRY_SCORE: float = field(default_factory=lambda: _float("STRONG_ENTRY_SCORE", 80.0))
    STRONG_ENTRY_CONFIDENCE: float = field(default_factory=lambda: _float("STRONG_ENTRY_CONFIDENCE", 0.80))
    CONFIRMATION_MIN_AGENTS: int = field(default_factory=lambda: _int("CONFIRMATION_MIN_AGENTS", 3))
    LIQUIDITY_MIN_SCORE: float = field(default_factory=lambda: _float("LIQUIDITY_MIN_SCORE", 0.60))
    EARLY_MOMENTUM_SCORE_MIN: float = field(default_factory=lambda: _float("EARLY_MOMENTUM_SCORE_MIN", 55.0))

    # Execution
    SLIPPAGE_PCT: float = field(default_factory=lambda: _float("SLIPPAGE_PCT", 0.0005))
    COMMISSION_PCT: float = field(default_factory=lambda: _float("COMMISSION_PCT", 0.001))
    ORDER_TIMEOUT_SECONDS: int = field(default_factory=lambda: _int("ORDER_TIMEOUT_SECONDS", 30))
    MAX_RETRIES: int = field(default_factory=lambda: _int("MAX_RETRIES", 3))

    # Scan interval
    SCAN_INTERVAL_MINUTES: int = field(default_factory=lambda: _int("SCAN_INTERVAL_MINUTES", 15))
    PAPER_TRADE_GATE_MIN_TRADES: int = field(default_factory=lambda: _int("PAPER_TRADE_GATE_MIN_TRADES", 50))

    # Live trading gates (both required)
    LIVE_TRADING_ENABLED: bool = field(default_factory=lambda: _bool("LIVE_TRADING_ENABLED", False))
    LIVE_TRADING_CONFIRMED: bool = field(default_factory=lambda: _bool("LIVE_TRADING_CONFIRMED", False))

    @property
    def live_trading_active(self) -> bool:
        return self.LIVE_TRADING_ENABLED and self.LIVE_TRADING_CONFIRMED


settings = Settings()
