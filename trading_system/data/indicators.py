"""Technical indicator calculations — pure pandas/numpy, no TA-Lib."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from trading_system.core.exceptions import InsufficientDataError
from trading_system.data.models import IndicatorSet


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    # When both gain and loss are 0 (flat series), RSI is 50 by convention
    both_zero = (gain == 0) & (loss == 0)
    rs = gain / loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result[both_zero] = 50.0
    return result


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """ADX with +DI and -DI using Wilder smoothing."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    alpha = 1 / period
    atr_s = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s.replace(0, np.nan)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_series = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx_series, plus_di, minus_di


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """ATR using Wilder smoothing."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger_bands(close: pd.Series, period: int = 20,
                    std_dev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: upper, mid, lower."""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def vwap(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series) -> pd.Series:
    """Volume-weighted average price (session cumulative)."""
    typical = (high + low + close) / 3
    return (typical * volume).cumsum() / volume.cumsum()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-balance volume."""
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()


def rvol(volume: pd.Series, period: int = 20) -> pd.Series:
    """Relative volume vs rolling average."""
    avg = volume.rolling(period).mean()
    return volume / avg.replace(0, np.nan)


def support_resistance_levels(
    high: pd.Series, low: pd.Series, close: pd.Series,
    lookback: int = 50,
) -> tuple[list[float], list[float]]:
    """
    Identify swing highs (resistance) and swing lows (support).
    Returns (resistance_levels, support_levels), each sorted ascending.
    """
    window = 5
    h = high.tail(lookback).reset_index(drop=True)
    l = low.tail(lookback).reset_index(drop=True)
    c = close.tail(lookback).reset_index(drop=True)

    resistances: list[float] = []
    supports: list[float] = []

    for i in range(window, len(c) - window):
        if all(h[i] >= h[j] for j in range(i - window, i + window + 1) if j != i):
            resistances.append(float(h[i]))
        if all(l[i] <= l[j] for j in range(i - window, i + window + 1) if j != i):
            supports.append(float(l[i]))

    def cluster(levels: list[float], threshold: float = 0.005) -> list[float]:
        if not levels:
            return []
        levels = sorted(set(levels))
        clustered = [levels[0]]
        for lvl in levels[1:]:
            if abs(lvl - clustered[-1]) / max(clustered[-1], 1e-9) > threshold:
                clustered.append(lvl)
        return clustered

    resistances = sorted(cluster(resistances))
    supports = sorted(cluster(supports))
    return resistances, supports


def _slope(series: pd.Series) -> float:
    """(current - 3_bars_ago) / 3_bars_ago slope."""
    if len(series) < 4:
        return 0.0
    prev = series.iloc[-4]
    curr = series.iloc[-1]
    if prev == 0:
        return 0.0
    return float((curr - prev) / abs(prev))


def compute_indicator_set(df: pd.DataFrame, timestamp: datetime) -> IndicatorSet:
    """Compute all indicators from OHLCV DataFrame. Requires >= 250 bars."""
    if len(df) < 250:
        raise InsufficientDataError(
            f"Need >= 250 bars, got {len(df)}",
            {"bars": len(df)},
        )

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    current_price = float(close.iloc[-1])

    ema20_s = ema(close, 20)
    ema50_s = ema(close, 50)
    ema200_s = ema(close, 200)
    vwap_s = vwap(high, low, close, volume)
    rsi_s = rsi(close, 14)
    macd_line_s, macd_sig_s, macd_hist_s = macd(close)
    adx_s, di_plus_s, di_minus_s = adx(high, low, close, 14)
    atr_s = atr(high, low, close, 14)
    bb_upper_s, bb_mid_s, bb_lower_s = bollinger_bands(close, 20, 2.0)
    obv_s = obv(close, volume)
    rvol_s = rvol(volume, 20)

    bb_width_s = (bb_upper_s - bb_lower_s) / bb_mid_s.replace(0, np.nan)
    bb_width_avg20 = float(bb_width_s.rolling(20).mean().iloc[-1])
    bb_squeeze_threshold = float(bb_width_s.rolling(100).quantile(0.20).iloc[-1])
    current_bb_width = float(bb_width_s.iloc[-1])
    bb_squeeze = current_bb_width < bb_squeeze_threshold

    current_atr = float(atr_s.iloc[-1])
    atr_avg20 = float(atr_s.rolling(20).mean().iloc[-1])
    atr_pct = (current_atr / current_price * 100) if current_price else 0.0

    current_vwap = float(vwap_s.iloc[-1])
    price_vs_vwap = (current_price - current_vwap) / current_vwap if current_vwap else 0.0

    obv_slope = _slope(obv_s)
    current_rvol = float(rvol_s.iloc[-1]) if not np.isnan(rvol_s.iloc[-1]) else 1.0

    rsi_prev = float(rsi_s.iloc[-4]) if len(rsi_s) >= 4 else float(rsi_s.iloc[-1])
    rsi_slope = _slope(rsi_s)

    res_levels, sup_levels = support_resistance_levels(high, low, close)

    def nearest_above(levels: list[float], price: float) -> float:
        above = [l for l in levels if l > price]
        return min(above) if above else price * 1.05

    def nearest_below(levels: list[float], price: float) -> float:
        below = [l for l in levels if l < price]
        return max(below) if below else price * 0.95

    return IndicatorSet(
        ema20=float(ema20_s.iloc[-1]),
        ema50=float(ema50_s.iloc[-1]),
        ema200=float(ema200_s.iloc[-1]),
        ema20_slope=_slope(ema20_s),
        ema50_slope=_slope(ema50_s),
        price_vs_vwap=price_vs_vwap,
        rsi=float(rsi_s.iloc[-1]),
        rsi_prev=rsi_prev,
        rsi_slope=rsi_slope,
        macd_line=float(macd_line_s.iloc[-1]),
        macd_signal=float(macd_sig_s.iloc[-1]),
        macd_histogram=float(macd_hist_s.iloc[-1]),
        macd_hist_slope=float(macd_hist_s.iloc[-1] - macd_hist_s.iloc[-2]),
        adx=float(adx_s.iloc[-1]) if not np.isnan(adx_s.iloc[-1]) else 0.0,
        di_plus=float(di_plus_s.iloc[-1]) if not np.isnan(di_plus_s.iloc[-1]) else 0.0,
        di_minus=float(di_minus_s.iloc[-1]) if not np.isnan(di_minus_s.iloc[-1]) else 0.0,
        atr=current_atr,
        atr_pct=atr_pct,
        atr_avg20=atr_avg20,
        bb_upper=float(bb_upper_s.iloc[-1]),
        bb_lower=float(bb_lower_s.iloc[-1]),
        bb_mid=float(bb_mid_s.iloc[-1]),
        bb_width=current_bb_width,
        bb_width_avg20=bb_width_avg20,
        bb_squeeze=bb_squeeze,
        obv=float(obv_s.iloc[-1]),
        obv_slope=obv_slope,
        rvol=current_rvol,
        vwap=current_vwap,
        resistance_levels=res_levels[:3],
        support_levels=sup_levels[-3:] if len(sup_levels) >= 3 else sup_levels,
        nearest_resistance=nearest_above(res_levels, current_price),
        nearest_support=nearest_below(sup_levels, current_price),
        timestamp=timestamp,
    )
