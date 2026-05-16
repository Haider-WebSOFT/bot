"""Multi-timeframe alignment scoring and counter-trend detection."""
from __future__ import annotations

from trading_system.data.models import MarketSnapshot


def compute_mtf_alignment(snap: MarketSnapshot) -> float:
    """
    Returns 0–100 alignment score.
    Higher = stronger institutional bull setup.
    """
    score = 0.0
    tf4 = snap.tf_4h
    tf1 = snap.tf_1h
    tf15 = snap.tf_15m

    # 4H macro structure (max 40)
    if tf4.ema20 > tf4.ema50 > tf4.ema200:
        score += 15
    if tf4.adx > 25:
        score += 15
    if tf4.ema20_slope > 0 and tf4.adx > 20:
        score += 10

    # 1H confirmation (max 35)
    if tf1.rsi > 50 and tf1.rsi_slope > 0:
        score += 15
    if tf1.macd_hist_slope > 0:
        score += 10
    if abs(tf1.price_vs_vwap) < 0.005:
        score += 10

    # 15m entry timing (max 25)
    if tf15.rvol > 1.5:
        score += 15
    if tf15.bb_squeeze:
        score += 10

    return min(score, 100.0)


def is_counter_trend(snap: MarketSnapshot) -> bool:
    """
    Returns True if 4H is in a downtrend — blocks long entry.
    """
    tf4 = snap.tf_4h
    bear_4h = tf4.ema20 < tf4.ema50 and tf4.ema50 < tf4.ema200
    return bear_4h
