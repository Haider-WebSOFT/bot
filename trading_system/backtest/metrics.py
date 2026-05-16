"""Backtest performance metrics computation."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from trading_system.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricsResult:
    """Complete performance metrics for a backtest or paper-trading period."""
    total_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    avg_rr: float
    expectancy: float
    profit_factor: float
    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_bars: int
    avg_trade_duration_bars: float
    regime_breakdown: dict[str, dict]
    monthly_returns: dict[str, float]
    mae_avg: float
    mfe_avg: float
    mae_mfe_ratio: float
    strategy_breakdown: dict[str, dict]   # "standard" vs "early_momentum"


def compute_metrics(
    trades: list[dict],
    equity_curve: list[float],
    initial_capital: float = 10.0,
    risk_free_annual: float = 0.04,
    bars_per_year: int = 35040,       # 15m bars in a year
) -> MetricsResult:
    """
    Compute full performance metrics from trade list and equity curve.
    Logs WARNING if suspicious results detected (possible overfitting).
    """
    if not trades:
        return _empty_metrics()

    wins = [t for t in trades if t.get("pnl_pct", 0) > 0]
    losses = [t for t in trades if t.get("pnl_pct", 0) <= 0]

    win_rate = len(wins) / len(trades)
    avg_win = float(np.mean([t["pnl_pct"] for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t["pnl_pct"] for t in losses])) if losses else 0.0
    avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    gross_profit = sum(t.get("pnl_dollar", 0) for t in wins)
    gross_loss = abs(sum(t.get("pnl_dollar", 0) for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    eq = np.array(equity_curve, dtype=float)
    total_return = (eq[-1] - eq[0]) / eq[0] if eq[0] > 0 else 0.0

    n_bars = len(eq)
    years = n_bars / bars_per_year if bars_per_year > 0 else 1.0
    cagr = ((eq[-1] / eq[0]) ** (1 / max(years, 1/365)) - 1) if eq[0] > 0 else 0.0

    # Drawdown
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1)
    max_dd = float(dd.max())

    # Drawdown duration
    in_dd = dd > 0
    max_dd_dur = 0
    cur = 0
    for v in in_dd:
        if v:
            cur += 1
            max_dd_dur = max(max_dd_dur, cur)
        else:
            cur = 0

    # Returns per bar for Sharpe/Sortino
    bar_returns = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1)
    rf_per_bar = risk_free_annual / bars_per_year
    excess = bar_returns - rf_per_bar
    sharpe = (float(excess.mean()) / float(excess.std() + 1e-9)) * np.sqrt(bars_per_year)

    downside = bar_returns[bar_returns < rf_per_bar] - rf_per_bar
    sortino_denom = float(np.std(downside)) if len(downside) > 0 else 1e-9
    sortino = (float(excess.mean()) / sortino_denom) * np.sqrt(bars_per_year)

    calmar = cagr / max_dd if max_dd > 0 else 0.0

    avg_dur = float(np.mean([t.get("duration_bars", 0) for t in trades]))

    # Regime breakdown
    regime_breakdown: dict[str, dict] = {}
    for t in trades:
        r = t.get("regime", "unknown")
        if r not in regime_breakdown:
            regime_breakdown[r] = {"trades": 0, "wins": 0, "pnl": 0.0}
        regime_breakdown[r]["trades"] += 1
        if t.get("pnl_pct", 0) > 0:
            regime_breakdown[r]["wins"] += 1
        regime_breakdown[r]["pnl"] += t.get("pnl_dollar", 0)

    # Strategy breakdown
    strategy_breakdown: dict[str, dict] = {}
    for st in ("standard", "early_momentum"):
        subset = [t for t in trades if t.get("strategy_type", "standard") == st]
        if not subset:
            strategy_breakdown[st] = {"trades": 0, "win_rate": 0.0, "expectancy": 0.0}
            continue
        sw = [t for t in subset if t.get("pnl_pct", 0) > 0]
        sl = [t for t in subset if t.get("pnl_pct", 0) <= 0]
        swr = len(sw) / len(subset)
        savg_win = float(np.mean([t["pnl_pct"] for t in sw])) if sw else 0.0
        savg_loss = float(np.mean([t["pnl_pct"] for t in sl])) if sl else 0.0
        strategy_breakdown[st] = {
            "trades": len(subset),
            "win_rate": swr,
            "expectancy": (swr * savg_win) + ((1 - swr) * savg_loss),
        }

    # Monthly returns (approximate from equity curve)
    bars_per_month = bars_per_year // 12
    monthly: dict[str, float] = {}
    for i in range(0, len(eq) - bars_per_month, bars_per_month):
        mo = f"M{i // bars_per_month + 1}"
        monthly[mo] = float((eq[i + bars_per_month] - eq[i]) / max(eq[i], 1e-9))

    mae_vals = [t.get("mae", 0.0) for t in trades]
    mfe_vals = [t.get("mfe", 0.0) for t in trades]
    mae_avg = float(np.mean(mae_vals))
    mfe_avg = float(np.mean(mfe_vals))
    mae_mfe_ratio = mae_avg / max(mfe_avg, 1e-9)

    # Overfitting warnings
    if total_trades := len(trades):
        if profit_factor > 3.0:
            logger.warning("Possible overfitting or insufficient sample size",
                           extra={"reason": "profit_factor > 3.0",
                                  "profit_factor": profit_factor})
        if sharpe > 3.0:
            logger.warning("Possible overfitting or insufficient sample size",
                           extra={"reason": "sharpe_ratio > 3.0",
                                  "sharpe_ratio": sharpe})
        if total_trades < 30:
            logger.warning("Possible overfitting or insufficient sample size",
                           extra={"reason": "total_trades < 30",
                                  "total_trades": total_trades})

    return MetricsResult(
        total_trades=len(trades),
        win_rate=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        avg_rr=avg_rr,
        expectancy=expectancy,
        profit_factor=profit_factor,
        total_return_pct=total_return,
        cagr_pct=cagr,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown_pct=max_dd,
        max_drawdown_duration_bars=max_dd_dur,
        avg_trade_duration_bars=avg_dur,
        regime_breakdown=regime_breakdown,
        monthly_returns=monthly,
        mae_avg=mae_avg,
        mfe_avg=mfe_avg,
        mae_mfe_ratio=mae_mfe_ratio,
        strategy_breakdown=strategy_breakdown,
    )


def _empty_metrics() -> MetricsResult:
    return MetricsResult(
        total_trades=0, win_rate=0.0, avg_win_pct=0.0, avg_loss_pct=0.0,
        avg_rr=0.0, expectancy=0.0, profit_factor=0.0, total_return_pct=0.0,
        cagr_pct=0.0, sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
        max_drawdown_pct=0.0, max_drawdown_duration_bars=0,
        avg_trade_duration_bars=0.0, regime_breakdown={}, monthly_returns={},
        mae_avg=0.0, mfe_avg=0.0, mae_mfe_ratio=0.0,
        strategy_breakdown={"standard": {"trades": 0, "win_rate": 0.0, "expectancy": 0.0},
                            "early_momentum": {"trades": 0, "win_rate": 0.0, "expectancy": 0.0}},
    )
