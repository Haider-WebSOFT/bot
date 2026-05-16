# Halal Trading System

A professional, institutional-grade trading research and execution framework.
**Halal-compliant: spot market only, LONG only, no margin, no futures, no shorting.**

> ⚠️ **PROFITABILITY DISCLAIMER**
> This system does NOT guarantee profit. Profitability is an unproven hypothesis
> until validated through backtesting, walk-forward analysis, out-of-sample testing,
> Monte Carlo testing, and at least 50–100 paper trades.
> If no valid edge is found, live trading is blocked automatically.

---

## 1. System Overview

- **Halal compliance**: spot only, LONG only, zero leverage, zero borrowing
- **Bear market / chaos**: system holds cash — never forces a trade
- **Three strategies**:
  - **Trend Following** — 4H bull structure + 1H RSI confirmation
  - **Breakout** — Bollinger squeeze release + volume surge
  - **Strategy C: Early Momentum** — RSI acceleration ≥ 8 pts + RVOL ≥ 2.0 + MACD confirm

---

## 2. Environment Setup

**Prerequisites**: WSL2 Ubuntu 22.04, Python 3.11+

```bash
# python-binance and alpaca-trade-api are already installed
pip install -r requirements.txt
```

---

## 3. Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

Key variables:

| Variable | Default | Notes |
|---|---|---|
| `BINANCE_TESTNET` | `true` | Always use testnet first |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` | Paper trading |
| `LIVE_TRADING_ENABLED` | `false` | Both flags required for live |
| `LIVE_TRADING_CONFIRMED` | `false` | Both flags required for live |

Starting capital of **$5–10** is fine for paper trading.

> **Note:** Stock OHLCV uses **yfinance**, not Alpaca data API.
> Alpaca free tier does not include SIP consolidated market data.

---

## 4. Running the System

```bash
# Paper trading
python -m trading_system.paper_trading.paper_engine

# Check live trading gate
python -m trading_system.analytics.gate

# Streamlit dashboard
streamlit run trading_system/dashboard/app.py

# Backtests
python -m trading_system.backtest.engine

# Walk-forward validation
python -m trading_system.backtest.walk_forward
```

---

## 5. Live Trading Gate (7 Hard Criteria)

**All 7 must pass** before live trading is allowed:

| Criterion | Required |
|---|---|
| Minimum trades | ≥ 50 |
| Profit factor | ≥ 1.30 |
| Sharpe ratio | ≥ 1.00 |
| Max drawdown | ≤ 15% |
| Win rate | ≥ 40% |
| Distinct trading regimes | ≥ 2 |
| Expectancy | ≥ 0.003 |

If any criterion fails: **"NO VALID EDGE FOUND — LIVE TRADING BLOCKED"**

---

## 6. Risk Management

| Limit | Value |
|---|---|
| Risk per trade | 1% of equity |
| Max single position | 5% of equity |
| Max portfolio exposure | 25% |
| Max concurrent trades | 6 |
| Correlation threshold | 0.70 (max 2 correlated) |
| Daily loss halt | 3% |
| Weekly loss → half sizing | 7% |
| Max drawdown halt | 15% |
| Consecutive losses → half sizing | 4 losses |
| ATR stop-loss multiplier | 1.5× |
| ATR take-profit multiplier | 3.0× |
| ATR trailing stop | 0.5× |

---

## 7. Agent Descriptions

| Agent | Measures |
|---|---|
| **TrendAgent** | 4H EMA bull stack, ADX strength, VWAP position |
| **MomentumAgent** | RSI bull zone (50–65), MACD histogram acceleration |
| **VolumeAgent** | RVOL vs average, OBV accumulation slope |
| **VolatilityAgent** | BB squeeze release, ATR vs average, chaos detection |
| **LiquidityAgent** | Order book depth, spread, symbol tier — hard blocks illiquid |
| **SentimentAgent** | Fear & Greed index (contrarian), VIX — secondary signal |
| **ExecutionQualityAgent** | Tracks slippage history and fill rates |
| **EarlyMomentumAgent** | Strategy C — RSI acceleration + volume spike + MACD |

---

## 8. Regime Behaviour

| Regime | Action |
|---|---|
| TRENDING_BULL | Full trading — all strategies active |
| **TRENDING_BEAR** | **HOLD CASH — no new longs (Halal compliance)** |
| RANGING | Mean reversion longs only |
| VOL_COMPRESSION | Breakout anticipation |
| **HIGH_VOL_CHAOS** | **HOLD CASH — preserve capital** |

---

## 9. Strategy C — Early Momentum

**Problem solved**: standard agents miss fast-moving setups because they
require full multi-timeframe confirmation. Strategy C catches moves earlier.

**Entry conditions** (all 3 required):
1. **RSI Acceleration** ≥ 8 points in last 3 bars, current RSI 45–65, was below 50
2. **Volume Spike** RVOL ≥ 2.0 AND OBV slope positive
3. **MACD Confirmation** histogram rising and near/above zero

**Score threshold**: 55 (vs standard 60) — faster entry
**Regime**: TRENDING_BULL only, 4H EMA must be bullish, no ATR chaos

---

## 10. WSL2-Specific Notes

- Run all commands from WSL2 terminal (not PowerShell/CMD)
- SQLite database stored at `~/bot/trading_system/trading.db`
- Streamlit accessible at `http://localhost:8501` from Windows browser
- Log files at `trading_system/logs/trading.log`

---

## 11. Emergency Procedures

```bash
# Manual emergency flatten (closes all positions)
python -m trading_system.live.emergency

# Read the halt report
cat trading_system/halt_report.json
```

**halt_report.json** contains:
- `timestamp`: when the halt occurred
- `reason`: what triggered it
- `positions_closed`: list of symbols that were closed

---

## 12. FAQ

**Q: Why no shorting?**
A: Halal compliance. Spot only, no margin, no leverage, no short selling, no borrowing.

**Q: Why yfinance for stocks instead of Alpaca data?**
A: Alpaca free tier does not include SIP consolidated market data.
yfinance provides free OHLCV data adequate for our research purposes.

**Q: Can I use this with $5?**
A: Yes. The system detects small account sizing and logs warnings,
but produces valid (though tiny) position sizes. Always start with paper trading.

**Q: When can I go live?**
A: Only after all 7 LiveTradingGate criteria pass AND you manually set
`LIVE_TRADING_ENABLED=true` AND `LIVE_TRADING_CONFIRMED=true`.
Code correctness ≠ trading profitability.

**Q: What if the backtest looks too good (Sharpe > 3)?**
A: The system will log: *"Possible overfitting or insufficient sample size"*
and you should redo walk-forward validation before trusting the result.
