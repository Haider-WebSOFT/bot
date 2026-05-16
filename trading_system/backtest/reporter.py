"""HTML backtest report generator."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from trading_system.backtest.metrics import MetricsResult
from trading_system.core.logger import get_logger

logger = get_logger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "reports"


def generate_report(
    metrics: MetricsResult,
    trades: list[dict],
    equity_curve: list[float],
    title: str = "Backtest Report",
) -> Path:
    """Generate an HTML performance report."""
    REPORTS_DIR.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"backtest_{ts}.html"

    strategy_rows = ""
    for st, data in metrics.strategy_breakdown.items():
        strategy_rows += (
            f"<tr><td>{st}</td><td>{data['trades']}</td>"
            f"<td>{data['win_rate']:.1%}</td>"
            f"<td>{data['expectancy']:.3f}</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html>
<head><title>{title}</title>
<style>
  body {{ font-family: sans-serif; margin: 2em; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 2em; }}
  th, td {{ border: 1px solid #ccc; padding: 8px; text-align: right; }}
  th {{ background: #f0f0f0; }}
  .warn {{ color: orange; font-weight: bold; }}
  .block {{ color: red; font-weight: bold; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p>Generated: {datetime.utcnow().isoformat()}Z</p>
<h2>&#9888;&#65039; Halal Compliance: SPOT LONG ONLY &mdash; No shorting, no margin, no leverage</h2>
<h2>Key Metrics</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Total Trades</td><td>{metrics.total_trades}</td></tr>
  <tr><td>Win Rate</td><td>{metrics.win_rate:.1%}</td></tr>
  <tr><td>Profit Factor</td><td class="{'warn' if metrics.profit_factor > 3 else ''}">{metrics.profit_factor:.2f}</td></tr>
  <tr><td>Expectancy</td><td>{metrics.expectancy:.4f}</td></tr>
  <tr><td>Sharpe Ratio</td><td class="{'warn' if metrics.sharpe_ratio > 3 else ''}">{metrics.sharpe_ratio:.2f}</td></tr>
  <tr><td>Sortino Ratio</td><td>{metrics.sortino_ratio:.2f}</td></tr>
  <tr><td>Calmar Ratio</td><td>{metrics.calmar_ratio:.2f}</td></tr>
  <tr><td>Max Drawdown</td><td>{metrics.max_drawdown_pct:.1%}</td></tr>
  <tr><td>Total Return</td><td>{metrics.total_return_pct:.1%}</td></tr>
  <tr><td>CAGR</td><td>{metrics.cagr_pct:.1%}</td></tr>
</table>
<h2>Strategy Breakdown (Standard vs Early Momentum)</h2>
<table>
  <tr><th>Strategy</th><th>Trades</th><th>Win Rate</th><th>Expectancy</th></tr>
  {strategy_rows}
</table>
<h2>Regime Breakdown</h2>
<table>
  <tr><th>Regime</th><th>Trades</th><th>Wins</th><th>PnL</th></tr>
  {''.join(f"<tr><td>{r}</td><td>{d['trades']}</td><td>{d['wins']}</td><td>{d['pnl']:.4f}</td></tr>" for r, d in metrics.regime_breakdown.items())}
</table>
<p><em>{'&#9888;&#65039; WARNING: Possible overfitting (PF>3 or Sharpe>3)' if metrics.profit_factor > 3 or metrics.sharpe_ratio > 3 else '&#10003; No overfitting flags'}</em></p>
</body>
</html>"""

    out_path.write_text(html)
    logger.info("Report generated", extra={"path": str(out_path)})
    return out_path
