# System Checklist

## ✓ Halal Compliance (6 Items)

- [ ] LONG only — no short positions ever placed
- [ ] Spot market only — no futures, no perpetuals
- [ ] No margin or leverage of any kind
- [ ] No borrowing to fund positions
- [ ] TRENDING_BEAR regime → hold cash, no new longs
- [ ] HIGH_VOL_CHAOS regime → hold cash, no new positions

---

## ✓ LiveTradingGate — 7 Criteria (All Must Pass)

| # | Criterion | Required Value | How to Check |
|---|---|---|---|
| 1 | Total trades | ≥ 50 | `journal.get_all_trades()` |
| 2 | Profit factor | ≥ 1.30 | `metrics.profit_factor` |
| 3 | Sharpe ratio | ≥ 1.00 | `metrics.sharpe_ratio` |
| 4 | Max drawdown | ≤ 15% | `metrics.max_drawdown_pct` |
| 5 | Win rate | ≥ 40% | `metrics.win_rate` |
| 6 | Distinct trading regimes | ≥ 2 (excluding BEAR/CHAOS) | from closed trades |
| 7 | Expectancy | ≥ 0.003 | `metrics.expectancy` |

---

## ✓ Risk Limits (from settings.py)

| Limit | Value |
|---|---|
| BASE_RISK_PCT | 1% per trade |
| MAX_POSITION_PCT | 5% of equity max |
| MAX_PORTFOLIO_EXPOSURE | 25% |
| MAX_CONCURRENT_TRADES | 6 |
| CORRELATION_THRESHOLD | 0.70 |
| MAX_CORRELATED_POSITIONS | 2 |
| DAILY_LOSS_LIMIT_PCT | 3% |
| WEEKLY_LOSS_LIMIT_PCT | 7% |
| MAX_DRAWDOWN_PCT | 15% |
| CONSECUTIVE_LOSS_LIMIT | 4 |
| ATR_SL_MULTIPLIER | 1.5× |
| ATR_TP_MULTIPLIER | 3.0× |
| ATR_TRAILING_MULTIPLIER | 0.5× |
| ATR_BREAKEVEN_TRIGGER | 1.0R |
| SLIPPAGE_PCT | 0.05% |
| COMMISSION_PCT | 0.1% |

---

## ✓ 3 Conditions That Trigger Size Reduction

1. **ATR > 3% of price** → position size × 0.70 (volatility adjustment)
2. **Consecutive losses ≥ 4** → position size × 0.50
3. **Weekly loss ≥ 7%** → position size × 0.50

These stack multiplicatively if multiple conditions are true.

---

## ✓ 4 Conditions That Trigger Trading Halt

1. **Daily loss ≥ 3%** → halt for rest of day
2. **Max drawdown ≥ 15%** → full halt + alert
3. **`trading_halted = True`** → no new orders allowed
4. **HealthCheckError** on startup → engine aborts before starting

---

## ✓ 2 Conditions That Trigger Emergency Flatten

1. **Drawdown ≥ 22.5%** (1.5× MAX_DRAWDOWN_PCT) → RiskMonitor triggers
2. **Consecutive losses ≥ 8** (2× CONSECUTIVE_LOSS_LIMIT) → RiskMonitor triggers

Emergency flatten: all open LONG positions closed at market, halt_report.json written.

---

## ✓ Reading halt_report.json

```json
{
  "timestamp": "2024-01-15T14:32:01Z",
  "reason": "Max drawdown 22.5% exceeded emergency threshold",
  "positions_closed": ["BTCUSDT", "ETHUSDT"]
}
```

- **timestamp**: UTC time of the halt
- **reason**: what triggered the flatten
- **positions_closed**: symbols that were sold

After reading: fix the underlying issue before restarting the engine.
To restart: delete halt_report.json and restart the engine.

---

## ✓ Daily Monitoring Checklist

- [ ] Check dashboard at `http://localhost:8501`
- [ ] Verify equity curve is not in sustained drawdown
- [ ] Review any new risk alerts in the Risk Panel
- [ ] Check open positions — all LONG, all have valid stops
- [ ] Review last 5 trades for unexpected behaviour
- [ ] Check trading.log for ERROR or CRITICAL entries
- [ ] Verify regime detection is sensible for current market
- [ ] Confirm no overfitting warnings in logs

---

## ✓ Strategy C (EarlyMomentumAgent) Activation Conditions

All 3 conditions **must be met simultaneously**:

| Condition | Requirement |
|---|---|
| **RSI Acceleration** | 15m RSI rose ≥ 8 points since previous reading, current RSI between 45–65, previous RSI was below 50 |
| **Volume Spike** | 15m RVOL ≥ 2.0 AND OBV slope is positive |
| **MACD Confirmation** | 15m MACD histogram slope > 0 AND histogram value > −0.001 |

**Guards** (must all clear):
- Regime = TRENDING_BULL
- 4H EMA20 > EMA50 (bull structure)
- 15m ATR not in chaos (atr_pct < atr_avg20 × 2.0)
- Liquidity score > 0.50

**Score threshold**: 55 (vs standard 60)
**Strategy type recorded**: `"early_momentum"` in trade journal
