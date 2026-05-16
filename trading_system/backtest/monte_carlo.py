"""Monte Carlo robustness testing via trade sequence shuffling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trading_system.backtest.metrics import MetricsResult, compute_metrics
from trading_system.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    n_simulations: int
    median_return: float
    p5_return: float
    p95_return: float
    median_max_dd: float
    p95_max_dd: float
    probability_positive: float
    ruin_probability: float   # fraction of sims ending below 50% of capital


class MonteCarloSimulator:
    """Shuffles trade sequences to test strategy robustness."""

    def run(
        self,
        trades: list[dict],
        initial_capital: float = 10.0,
        n_simulations: int = 1000,
        ruin_threshold: float = 0.50,
    ) -> MonteCarloResult:
        if not trades:
            return MonteCarloResult(
                n_simulations=0, median_return=0.0, p5_return=0.0, p95_return=0.0,
                median_max_dd=0.0, p95_max_dd=0.0,
                probability_positive=0.0, ruin_probability=1.0,
            )

        pnl_pcts = np.array([t.get("pnl_pct", 0.0) for t in trades])
        rng = np.random.default_rng(42)
        returns: list[float] = []
        max_dds: list[float] = []
        ruins = 0

        for _ in range(n_simulations):
            shuffled = rng.choice(pnl_pcts, size=len(pnl_pcts), replace=True)
            equity = initial_capital
            peak = equity
            max_dd = 0.0
            for r in shuffled:
                equity *= (1 + r)
                peak = max(peak, equity)
                dd = (peak - equity) / peak if peak > 0 else 0.0
                max_dd = max(max_dd, dd)

            final_return = (equity - initial_capital) / initial_capital
            returns.append(final_return)
            max_dds.append(max_dd)
            if equity < initial_capital * ruin_threshold:
                ruins += 1

        returns_arr = np.array(returns)
        dds_arr = np.array(max_dds)

        return MonteCarloResult(
            n_simulations=n_simulations,
            median_return=float(np.median(returns_arr)),
            p5_return=float(np.percentile(returns_arr, 5)),
            p95_return=float(np.percentile(returns_arr, 95)),
            median_max_dd=float(np.median(dds_arr)),
            p95_max_dd=float(np.percentile(dds_arr, 95)),
            probability_positive=float((returns_arr > 0).mean()),
            ruin_probability=ruins / n_simulations,
        )
