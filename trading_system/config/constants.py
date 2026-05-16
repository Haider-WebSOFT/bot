"""System-wide constants and enumerations."""
from enum import Enum


class Regime(str, Enum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    VOL_COMPRESSION = "vol_compression"
    HIGH_VOL_CHAOS = "high_vol_chaos"


class Direction(str, Enum):
    LONG = "long"


class Strategy(str, Enum):
    TREND_FOLLOWING = "trend_following"
    MOMENTUM_CONTINUATION = "momentum_continuation"
    BREAKOUT = "breakout"
    MEAN_REVERSION = "mean_reversion"
    SUPPORT_RESISTANCE_BOUNCE = "support_resistance_bounce"
    BREAKOUT_ANTICIPATION = "breakout_anticipation"
    EARLY_MOMENTUM = "early_momentum"


class ExitReason(str, Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    BREAK_EVEN = "break_even"
    PARTIAL_EXIT = "partial_exit"
    TIME_EXIT = "time_exit"
    MOMENTUM_EXIT = "momentum_exit"
    RISK_HALT = "risk_halt"
    EMERGENCY_FLATTEN = "emergency_flatten"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


REGIME_RISK_MULTIPLIERS: dict[Regime, float] = {
    Regime.TRENDING_BULL:   1.0,
    Regime.TRENDING_BEAR:   0.0,
    Regime.RANGING:         0.7,
    Regime.VOL_COMPRESSION: 0.6,
    Regime.HIGH_VOL_CHAOS:  0.0,
}

REGIME_ALLOWED_STRATEGIES: dict[Regime, list[str]] = {
    Regime.TRENDING_BULL:   ["trend_following", "momentum_continuation",
                             "breakout", "early_momentum"],
    Regime.TRENDING_BEAR:   [],
    Regime.RANGING:         ["mean_reversion", "support_resistance_bounce"],
    Regime.VOL_COMPRESSION: ["breakout_anticipation"],
    Regime.HIGH_VOL_CHAOS:  [],
}

CRYPTO_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "LINKUSDT", "AVAXUSDT", "SUIUSDT", "MATICUSDT",
    "LTCUSDT", "BNBUSDT"
]

TIMEFRAMES = ["15m", "1h", "4h"]
