"""Module 3 indicator tests."""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 1.0)
    high = close + rng.uniform(0.1, 0.5, n)
    low = close - rng.uniform(0.1, 0.5, n)
    volume = rng.uniform(1000, 5000, n)
    return pd.DataFrame({
        "open": close - 0.1,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestEMA:
    def test_ema3_convergence(self):
        from trading_system.data.indicators import ema

        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ema(series, 3)
        assert len(result) == 5
        assert result.iloc[-1] > result.iloc[0]

    def test_ema_monotone_increasing_data(self):
        from trading_system.data.indicators import ema

        series = pd.Series(range(1, 21), dtype=float)
        result = ema(series, 5)
        assert result.iloc[-1] > result.iloc[5]


class TestRSI:
    def test_rsi_within_bounds(self):
        from trading_system.data.indicators import rsi

        df = _make_df()
        result = rsi(df["close"], 14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_flat_series(self):
        from trading_system.data.indicators import rsi

        flat = pd.Series([50.0] * 30)
        result = rsi(flat, 14)
        assert result.dropna().iloc[-1] == pytest.approx(50.0, abs=5.0)


class TestATR:
    def test_atr_positive(self):
        from trading_system.data.indicators import atr

        df = _make_df()
        result = atr(df["high"], df["low"], df["close"], 14)
        assert (result.dropna() > 0).all()

    def test_atr_manual_single_bar(self):
        from trading_system.data.indicators import atr

        high = pd.Series([10.0, 12.0])
        low = pd.Series([8.0, 9.0])
        close = pd.Series([9.0, 11.0])
        result = atr(high, low, close, 1)
        assert result.iloc[-1] > 0


class TestSupportResistance:
    def test_obvious_peaks_detected(self):
        from trading_system.data.indicators import support_resistance_levels

        n = 100
        close = pd.Series([100.0] * n)
        high = pd.Series([100.0] * n)
        low = pd.Series([100.0] * n)
        high.iloc[30] = 110.0
        low.iloc[70] = 90.0

        resistances, supports = support_resistance_levels(high, low, close, lookback=100)
        assert len(resistances) > 0 or len(supports) > 0

    def test_returns_sorted_lists(self):
        from trading_system.data.indicators import support_resistance_levels

        df = _make_df(200)
        resistances, supports = support_resistance_levels(
            df["high"], df["low"], df["close"]
        )
        assert resistances == sorted(resistances)
        assert supports == sorted(supports)


class TestComputeIndicatorSet:
    def test_insufficient_data_raises(self):
        from trading_system.data.indicators import compute_indicator_set
        from trading_system.core.exceptions import InsufficientDataError

        df = _make_df(100)
        with pytest.raises(InsufficientDataError):
            compute_indicator_set(df, datetime.utcnow())

    def test_full_compute_returns_indicator_set(self):
        from trading_system.data.indicators import compute_indicator_set
        from trading_system.data.models import IndicatorSet

        df = _make_df(300)
        result = compute_indicator_set(df, datetime.utcnow())
        assert isinstance(result, IndicatorSet)
        assert result.ema20 > 0
        assert 0 <= result.rsi <= 100
        assert result.atr > 0
        assert isinstance(result.bb_squeeze, bool)
