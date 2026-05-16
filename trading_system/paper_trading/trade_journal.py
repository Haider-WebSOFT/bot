"""SQLite trade journal via SQLAlchemy ORM."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, Float, Integer, String, Text, create_engine, DateTime,
)
from sqlalchemy.orm import DeclarativeBase, Session

from trading_system.core.logger import get_logger

logger = get_logger(__name__)

DB_PATH = Path(__file__).parent.parent / "trading.db"


class Base(DeclarativeBase):
    pass


class TradeRecord(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    asset_type = Column(String, default="crypto")
    direction = Column(String, default="long")   # always "long"
    strategy_type = Column(String, default="standard")
    entry_time = Column(DateTime)
    entry_price = Column(Float)
    exit_time = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    size = Column(Float)
    pnl_dollar = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    pnl_r = Column(Float, nullable=True)
    regime = Column(String, nullable=True)
    ensemble_score = Column(Float, nullable=True)
    agent_scores_json = Column(Text, nullable=True)
    exit_reason = Column(String, nullable=True)
    slippage_actual = Column(Float, nullable=True)
    mae = Column(Float, default=0.0)
    mfe = Column(Float, default=0.0)
    commission_paid = Column(Float, default=0.0)
    is_partial = Column(Boolean, default=False)


class EquitySnapshot(Base):
    __tablename__ = "equity_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    equity = Column(Float)
    drawdown_pct = Column(Float)


class ScanCycle(Base):
    __tablename__ = "scan_cycles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbols_scanned = Column(Integer)
    opportunities_found = Column(Integer)
    regime_distribution_json = Column(Text)


class TradeJournal:
    """SQLite-backed trade journal."""

    def __init__(self, db_path: str | None = None) -> None:
        path = db_path or str(DB_PATH)
        self._engine = create_engine(f"sqlite:///{path}", echo=False)
        Base.metadata.create_all(self._engine)

    def open_trade(self, card, sizing, actual_entry_price: float) -> int:
        """Record a new LONG trade entry. Returns trade ID."""
        with Session(self._engine) as sess:
            record = TradeRecord(
                symbol=card.symbol,
                asset_type=card.asset_type,
                direction="long",
                strategy_type=card.strategy_type,
                entry_time=card.timestamp,
                entry_price=actual_entry_price,
                size=sizing.quantity,
                regime=card.regime.value if hasattr(card.regime, "value") else str(card.regime),
                ensemble_score=card.ensemble_score,
                agent_scores_json=json.dumps(card.agent_scores),
                commission_paid=actual_entry_price * sizing.quantity * 0.001,
            )
            sess.add(record)
            sess.commit()
            trade_id = record.id
        logger.info("Trade opened", extra={"symbol": card.symbol, "trade_id": trade_id})
        return trade_id

    def close_trade(self, trade_id: int, exit_price: float,
                     exit_reason: str, mae: float, mfe: float) -> None:
        """Record trade exit."""
        with Session(self._engine) as sess:
            record = sess.get(TradeRecord, trade_id)
            if not record:
                return
            record.exit_price = exit_price
            record.exit_time = datetime.utcnow()
            record.exit_reason = exit_reason
            record.mae = mae
            record.mfe = mfe
            if record.entry_price and record.size:
                record.pnl_dollar = (exit_price - record.entry_price) * record.size
                record.pnl_pct = (exit_price - record.entry_price) / record.entry_price
                sl_dist = record.entry_price * 0.015
                record.pnl_r = (record.pnl_dollar / (sl_dist * record.size)
                                if sl_dist else 0.0)
            sess.commit()
        logger.info("Trade closed",
                    extra={"trade_id": trade_id, "exit_reason": exit_reason})

    def update_equity_snapshot(self, equity: float, drawdown: float) -> None:
        with Session(self._engine) as sess:
            snap = EquitySnapshot(equity=equity, drawdown_pct=drawdown)
            sess.add(snap)
            sess.commit()

    def get_all_trades(self) -> list[dict]:
        with Session(self._engine) as sess:
            records = sess.query(TradeRecord).all()
            return [self._to_dict(r) for r in records]

    def get_open_trades(self) -> list[dict]:
        with Session(self._engine) as sess:
            records = sess.query(TradeRecord).filter(
                TradeRecord.exit_price.is_(None)
            ).all()
            return [self._to_dict(r) for r in records]

    def get_last_n_trades(self, n: int) -> list[dict]:
        with Session(self._engine) as sess:
            records = (
                sess.query(TradeRecord)
                .order_by(TradeRecord.id.desc())
                .limit(n)
                .all()
            )
            return [self._to_dict(r) for r in reversed(records)]

    def _to_dict(self, r: TradeRecord) -> dict:
        return {
            "id": r.id, "symbol": r.symbol, "asset_type": r.asset_type,
            "direction": r.direction, "strategy_type": r.strategy_type,
            "entry_time": r.entry_time, "entry_price": r.entry_price,
            "exit_time": r.exit_time, "exit_price": r.exit_price,
            "size": r.size, "pnl_dollar": r.pnl_dollar, "pnl_pct": r.pnl_pct,
            "pnl_r": r.pnl_r, "regime": r.regime,
            "ensemble_score": r.ensemble_score,
            "agent_scores": json.loads(r.agent_scores_json or "{}"),
            "exit_reason": r.exit_reason, "slippage_actual": r.slippage_actual,
            "mae": r.mae, "mfe": r.mfe, "commission_paid": r.commission_paid,
            "is_partial": r.is_partial,
        }
