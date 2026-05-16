"""Data models for market snapshots and indicator sets."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class IndicatorSet:
    """Computed indicators for a single symbol/timeframe."""

    # Price structure
    ema20: float
    ema50: float
    ema200: float
    ema20_slope: float
    ema50_slope: float
    price_vs_vwap: float

    # Momentum
    rsi: float
    rsi_prev: float
    rsi_slope: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    macd_hist_slope: float

    # Trend strength
    adx: float
    di_plus: float
    di_minus: float

    # Volatility
    atr: float
    atr_pct: float
    atr_avg20: float
    bb_upper: float
    bb_lower: float
    bb_mid: float
    bb_width: float
    bb_width_avg20: float
    bb_squeeze: bool

    # Volume
    obv: float
    obv_slope: float
    rvol: float
    vwap: float

    # Support / Resistance
    resistance_levels: list[float]
    support_levels: list[float]
    nearest_resistance: float
    nearest_support: float

    timestamp: datetime


@dataclass(slots=True)
class MarketSnapshot:
    """Full multi-timeframe snapshot for a symbol."""

    symbol: str
    current_price: float
    tf_15m: IndicatorSet
    tf_1h: IndicatorSet
    tf_4h: IndicatorSet
    mtf_alignment_score: float
    spread_pct: float
    liquidity_score: float
    timestamp: datetime
