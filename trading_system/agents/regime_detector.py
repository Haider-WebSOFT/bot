"""Rule-based market regime detector."""
from __future__ import annotations

import sqlite3
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from trading_system.config.constants import (
    Regime, REGIME_ALLOWED_STRATEGIES, REGIME_RISK_MULTIPLIERS,
)
from trading_system.core.logger import get_logger
from trading_system.data.models import MarketSnapshot

logger = get_logger(__name__)


@dataclass
class RegimeResult:
    """Output of the regime detector for a single snapshot."""

    regime: Regime
    confidence: float
    risk_multiplier: float
    allowed_strategies: list[str]
    regime_scores: dict[str, float]
    reasoning: dict[str, str]
    timestamp: datetime


class RegimeDetector:
    """Deterministic rule-based market regime classifier."""

    def __init__(self) -> None:
        self._history: deque[Regime] = deque(maxlen=20)

    def detect(self, snap: MarketSnapshot) -> RegimeResult:
        """Classify the current regime from a MarketSnapshot."""
        tf4 = snap.tf_4h
        tf1 = snap.tf_1h
        tf15 = snap.tf_15m
        price = snap.current_price
        scores: dict[str, float] = {}
        reasoning: dict[str, str] = {}

        # 1. HIGH_VOL_CHAOS — overrides everything
        chaos_score = 0.0
        if tf15.atr_pct > tf15.atr_avg20 * 2.0:
            chaos_score = 1.0
            reasoning["chaos_15m"] = f"15m ATR {tf15.atr_pct:.2f}% > 2x avg"
        elif tf4.atr_pct > tf4.atr_avg20 * 2.0:
            chaos_score = 1.0
            reasoning["chaos_4h"] = f"4H ATR {tf4.atr_pct:.2f}% > 2x avg"
        scores[Regime.HIGH_VOL_CHAOS] = chaos_score

        if chaos_score >= 1.0:
            regime = Regime.HIGH_VOL_CHAOS
            return self._build_result(regime, 1.0, scores, reasoning, snap.timestamp)

        # 2. TRENDING_BULL
        bull_score = 0.0
        if tf4.adx > 25:
            bull_score += 0.30; reasoning["bull_adx"] = f"ADX={tf4.adx:.1f}"
        if tf4.ema20 > tf4.ema50:
            bull_score += 0.20; reasoning["bull_ema20>50"] = "ema20>ema50"
        if tf4.ema50 > tf4.ema200:
            bull_score += 0.20; reasoning["bull_ema50>200"] = "ema50>ema200"
        if price > tf4.ema20:
            bull_score += 0.15; reasoning["bull_price>ema20"] = "price above ema20"
        if tf1.rsi > 50:
            bull_score += 0.15; reasoning["bull_rsi"] = f"1H RSI={tf1.rsi:.1f}"
        scores[Regime.TRENDING_BULL] = bull_score

        # 3. TRENDING_BEAR
        bear_score = 0.0
        if tf4.adx > 25:
            bear_score += 0.30
        if tf4.ema20 < tf4.ema50:
            bear_score += 0.20; reasoning["bear_ema20<50"] = "ema20<ema50"
        if tf4.ema50 < tf4.ema200:
            bear_score += 0.20; reasoning["bear_ema50<200"] = "ema50<ema200"
        if price < tf4.ema20:
            bear_score += 0.15
        if tf1.rsi < 50:
            bear_score += 0.15
        scores[Regime.TRENDING_BEAR] = bear_score

        # 4. RANGING
        ranging_score = 0.0
        if tf4.adx < 20:
            ranging_score += 0.40; reasoning["ranging_adx"] = f"ADX={tf4.adx:.1f} weak"
        if tf4.bb_width < tf4.bb_width_avg20 and not tf15.bb_squeeze:
            ranging_score += 0.30
        if tf15.rvol < 1.2 and not tf15.bb_squeeze:
            ranging_score += 0.30
        scores[Regime.RANGING] = ranging_score

        # 5. VOL_COMPRESSION
        compression_score = 0.0
        if tf15.bb_squeeze:
            compression_score += 0.50; reasoning["compression_squeeze"] = "BB squeeze"
        if tf4.atr_pct < tf4.atr_avg20 * 0.8:
            compression_score += 0.30
        if tf4.adx < 20:
            compression_score += 0.20
        scores[Regime.VOL_COMPRESSION] = compression_score

        # Pick highest non-chaos score
        non_chaos = {k: v for k, v in scores.items() if k != Regime.HIGH_VOL_CHAOS}
        regime = max(non_chaos, key=lambda r: non_chaos[r])
        confidence = non_chaos[regime]

        # Stability adjustment
        self._history.append(regime)
        if len(self._history) >= 10:
            stability = sum(1 for r in self._history if r == regime) / len(self._history)
            if stability < 0.6:
                confidence *= 0.80
                reasoning["stability"] = f"Low regime stability ({stability:.0%})"

        return self._build_result(regime, confidence, scores, reasoning, snap.timestamp)

    def _build_result(self, regime: Regime, confidence: float,
                      scores: dict, reasoning: dict,
                      timestamp: datetime) -> RegimeResult:
        return RegimeResult(
            regime=regime,
            confidence=min(confidence, 1.0),
            risk_multiplier=REGIME_RISK_MULTIPLIERS[regime],
            allowed_strategies=list(REGIME_ALLOWED_STRATEGIES[regime]),
            regime_scores={r.value: s for r, s in scores.items()},
            reasoning=reasoning,
            timestamp=timestamp,
        )


class RegimeHistory:
    """Persists regime readings to SQLite and memory."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "trading.db")
        self._db_path = db_path
        self._memory: deque[RegimeResult] = deque(maxlen=100)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                regime TEXT,
                confidence REAL,
                risk_multiplier REAL
            )
        """)
        conn.commit()
        conn.close()

    def record(self, result: RegimeResult) -> None:
        self._memory.append(result)
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "INSERT INTO regime_history (timestamp, regime, confidence, risk_multiplier)"
            " VALUES (?, ?, ?, ?)",
            (result.timestamp.isoformat(), result.regime.value,
             result.confidence, result.risk_multiplier),
        )
        conn.commit()
        conn.close()

    def get_current_regime(self) -> RegimeResult | None:
        return self._memory[-1] if self._memory else None

    def get_stability_score(self) -> float:
        if not self._memory:
            return 0.0
        current = self._memory[-1].regime
        return sum(1 for r in self._memory if r.regime == current) / len(self._memory)

    def regime_changed_recently(self, bars: int = 3) -> bool:
        if len(self._memory) < bars + 1:
            return False
        recent = list(self._memory)[-bars - 1:]
        return len(set(r.regime for r in recent)) > 1


def label_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime labels bar-by-bar (no lookahead). Used by backtester."""
    from trading_system.data.models import MarketSnapshot
    from trading_system.data.feed import _empty_indicator_set

    detector = RegimeDetector()
    regimes = []
    confidences = []
    ts = datetime.utcnow()

    for i in range(len(df)):
        sub = df.iloc[:i + 1]
        if len(sub) < 50:
            regimes.append(Regime.RANGING.value)
            confidences.append(0.5)
            continue

        ind = _empty_indicator_set(sub, ts)
        snap = MarketSnapshot(
            symbol="X", current_price=float(sub["close"].iloc[-1]),
            tf_15m=ind, tf_1h=ind, tf_4h=ind,
            mtf_alignment_score=0.0, spread_pct=0.001,
            liquidity_score=0.8, timestamp=ts,
        )
        result = detector.detect(snap)
        regimes.append(result.regime.value)
        confidences.append(result.confidence)

    df = df.copy()
    df["regime"] = regimes
    df["regime_confidence"] = confidences
    return df
