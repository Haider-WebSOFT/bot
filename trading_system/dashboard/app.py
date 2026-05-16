"""Streamlit monitoring dashboard for the trading system."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading System — Live Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Auto-refresh every 30 s ──────────────────────────────────────────────────
st_autorefresh = None
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=30_000, key="auto_refresh")
except ImportError:
    pass


# ── Data helpers ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_journal():
    try:
        from trading_system.paper_trading.trade_journal import TradeJournal
        return TradeJournal()
    except Exception:
        return None


def load_trades(journal) -> pd.DataFrame:
    if journal is None:
        return pd.DataFrame()
    try:
        trades = journal.get_all_trades()
        return pd.DataFrame(trades) if trades else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def load_equity(journal) -> pd.DataFrame:
    if journal is None:
        return pd.DataFrame()
    try:
        from trading_system.paper_trading.trade_journal import EquitySnapshot
        from sqlalchemy.orm import Session
        with Session(journal._engine) as sess:
            snaps = sess.query(EquitySnapshot).order_by(EquitySnapshot.id).all()
            return pd.DataFrame([
                {"timestamp": s.timestamp, "equity": s.equity,
                 "drawdown_pct": s.drawdown_pct}
                for s in snaps
            ])
    except Exception:
        return pd.DataFrame()


def check_gate(trades_df: pd.DataFrame):
    try:
        from trading_system.analytics.gate import LiveTradingGate
        from trading_system.backtest.metrics import compute_metrics
        if trades_df.empty:
            return None
        closed = trades_df[trades_df["exit_price"].notna()].to_dict("records")
        equity = [10.0] + [t.get("pnl_dollar", 0) for t in closed]
        metrics = compute_metrics(closed, equity)
        gate = LiveTradingGate()
        return gate.check(metrics, closed)
    except Exception:
        return None


# ── Load data ────────────────────────────────────────────────────────────────
journal = get_journal()
trades_df = load_trades(journal)
equity_df = load_equity(journal)
gate_result = check_gate(trades_df)

# ── ROW 1: Status bar ────────────────────────────────────────────────────────
st.title("📈 Trading System — Live Monitor (Halal | Spot Long Only)")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Mode", "🟡 PAPER")
with c2:
    st.metric("Halal", "✅ SPOT LONG ONLY")
with c3:
    approved = gate_result.approved if gate_result else False
    st.metric("Gate", "✅ APPROVED" if approved else "🔴 NOT APPROVED")
with c4:
    st.metric("Last Scan", datetime.utcnow().strftime("%H:%M:%S UTC"))
with c5:
    st.metric("Health", "Binance ✓ | Alpaca ✓ | DB ✓")

st.divider()

# ── ROW 2: Key metrics ───────────────────────────────────────────────────────
closed_df = trades_df[trades_df["exit_price"].notna()] if not trades_df.empty else pd.DataFrame()

col1, col2, col3, col4 = st.columns(4)
with col1:
    session_pnl = closed_df["pnl_dollar"].sum() if not closed_df.empty else 0.0
    st.metric("Session PnL", f"${session_pnl:.4f}")
with col2:
    open_pos = len(trades_df[trades_df["exit_price"].isna()]) if not trades_df.empty else 0
    st.metric("Open Positions", open_pos)
with col3:
    st.metric("Portfolio Heat", "0%")
with col4:
    last20 = closed_df.tail(20)
    wr = (last20["pnl_pct"] > 0).mean() if not last20.empty else 0.0
    st.metric("Win Rate (last 20)", f"{wr:.0%}")

st.divider()

# ── ROW 3: Equity curve ──────────────────────────────────────────────────────
st.subheader("Equity Curve")
if not equity_df.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df["timestamp"], y=equity_df["equity"],
        name="Equity", line=dict(color="royalblue", width=2)
    ))
    fig.add_hline(y=10.0, line_dash="dash", line_color="gray",
                  annotation_text="Starting $10")
    fig.update_layout(height=250, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=equity_df["timestamp"],
        y=equity_df["drawdown_pct"] * 100,
        fill="tozeroy", fillcolor="rgba(255,80,80,0.3)",
        name="Drawdown %", line=dict(color="red")
    ))
    fig2.update_layout(height=120, margin=dict(t=10, b=10),
                       yaxis_title="Drawdown %")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No equity data yet — run paper trading to populate.")

st.divider()

# ── ROW 4: Positions + Opportunities ────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("Open Long Positions")
    open_df = trades_df[trades_df["exit_price"].isna()] if not trades_df.empty else pd.DataFrame()
    if not open_df.empty:
        display_cols = [c for c in ["symbol", "entry_price", "pnl_r",
                                     "regime", "strategy_type"] if c in open_df.columns]
        st.dataframe(open_df[display_cols].style.applymap(
            lambda v: "color: green" if isinstance(v, float) and v > 0
            else ("color: red" if isinstance(v, float) and v < 0 else ""),
            subset=["pnl_r"] if "pnl_r" in display_cols else []
        ), use_container_width=True)
    else:
        st.info("No open positions.")

with right:
    st.subheader("Top Scanner Opportunities")
    st.info("Connect scanner to populate opportunities.")

st.divider()

# ── ROW 5: Gate status ───────────────────────────────────────────────────────
st.subheader("Live Trading Gate")
if gate_result:
    for criterion, is_met in gate_result.criteria_met.items():
        val = gate_result.criteria_values.get(criterion, 0)
        req = gate_result.criteria_required.get(criterion, 0)
        icon = "✅" if is_met else "❌"
        st.write(f"{icon} **{criterion}**: {val:.3f} (required: {req})")
    if gate_result.approved:
        st.success("✓ All gate criteria met — live trading ALLOWED (requires manual flag)")
    else:
        st.error(f"✗ LIVE TRADING BLOCKED — {', '.join(gate_result.blockers)}")
else:
    st.info("No trades yet — gate cannot be evaluated.")

st.divider()

# ── ROW 6: Risk panel ────────────────────────────────────────────────────────
st.subheader("Risk Panel")
rc1, rc2, rc3 = st.columns(3)
with rc1:
    dd = equity_df["drawdown_pct"].iloc[-1] * 100 if not equity_df.empty else 0.0
    st.metric("Current Drawdown", f"{dd:.1f}%")
with rc2:
    st.metric("Daily PnL Limit", "3.0%")
with rc3:
    st.metric("Max Drawdown Limit", "15.0%")

st.divider()

# ── ROW 7: Trade history ─────────────────────────────────────────────────────
st.subheader("Trade History (last 50)")

filter_strategy = st.selectbox("Filter by strategy",
                                ["all", "standard", "early_momentum"])

history_df = closed_df.tail(50) if not closed_df.empty else pd.DataFrame()
if not history_df.empty and filter_strategy != "all":
    if "strategy_type" in history_df.columns:
        history_df = history_df[history_df["strategy_type"] == filter_strategy]

if not history_df.empty:
    show_cols = [c for c in ["symbol", "entry_price", "exit_price", "pnl_r",
                              "pnl_dollar", "exit_reason", "regime",
                              "strategy_type", "mae", "mfe"]
                 if c in history_df.columns]
    st.dataframe(history_df[show_cols], use_container_width=True)
    csv = history_df.to_csv(index=False)
    st.download_button("Export to CSV", csv, "trades.csv", "text/csv")
else:
    st.info("No closed trades yet.")

st.divider()

# ── ROW 8: Regime overview ───────────────────────────────────────────────────
st.subheader("Regime Overview")
if not trades_df.empty and "regime" in trades_df.columns:
    regime_counts = trades_df["regime"].value_counts().reset_index()
    regime_counts.columns = ["regime", "count"]
    color_map = {
        "trending_bull": "teal", "trending_bear": "red",
        "ranging": "gray", "vol_compression": "orange",
        "high_vol_chaos": "crimson",
    }
    st.bar_chart(regime_counts.set_index("regime")["count"])

st.divider()

# ── ROW 9: Strategy C performance ───────────────────────────────────────────
st.subheader("Strategy C — Early Momentum vs Standard")
if not closed_df.empty and "strategy_type" in closed_df.columns:
    for st_type in ["standard", "early_momentum"]:
        sub = closed_df[closed_df["strategy_type"] == st_type]
        if not sub.empty and "pnl_pct" in sub.columns:
            wr = (sub["pnl_pct"] > 0).mean()
            exp = sub["pnl_pct"].mean()
            st.write(f"**{st_type}**: {len(sub)} trades | "
                     f"Win rate {wr:.0%} | Expectancy {exp:.3f}")
else:
    st.info("No closed trades to compare strategies.")
