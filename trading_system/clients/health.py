"""System health checker."""
import time
from dataclasses import dataclass, field
from datetime import datetime

from trading_system.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HealthReport:
    status: str
    binance_ok: bool
    alpaca_ok: bool
    database_ok: bool
    latency_ms: dict[str, float]
    errors: list[str]
    timestamp: datetime


class HealthChecker:
    """Checks connectivity and latency for all system components."""

    async def check_all(self, binance, alpaca) -> HealthReport:
        """Ping exchanges and database, return HealthReport."""
        errors: list[str] = []
        latency: dict[str, float] = {}

        binance_ok = False
        try:
            t0 = time.monotonic()
            await binance.get_account_balance()
            latency["binance_ms"] = (time.monotonic() - t0) * 1000
            binance_ok = True
        except Exception as exc:
            errors.append(f"Binance: {exc}")
            latency["binance_ms"] = -1.0
            logger.error("Binance health check failed", extra={"error": str(exc)})

        alpaca_ok = False
        try:
            t0 = time.monotonic()
            await alpaca.get_account()
            latency["alpaca_ms"] = (time.monotonic() - t0) * 1000
            alpaca_ok = True
        except Exception as exc:
            errors.append(f"Alpaca: {exc}")
            latency["alpaca_ms"] = -1.0
            logger.error("Alpaca health check failed", extra={"error": str(exc)})

        database_ok = False
        try:
            import sqlite3
            from pathlib import Path
            db_path = Path(__file__).parent.parent / "trading.db"
            t0 = time.monotonic()
            conn = sqlite3.connect(str(db_path))
            conn.execute("SELECT 1")
            conn.close()
            latency["database_ms"] = (time.monotonic() - t0) * 1000
            database_ok = True
        except Exception as exc:
            errors.append(f"Database: {exc}")
            latency["database_ms"] = -1.0

        if not binance_ok or not alpaca_ok:
            status = "critical"
        elif not database_ok:
            status = "degraded"
        else:
            status = "healthy"

        report = HealthReport(
            status=status,
            binance_ok=binance_ok,
            alpaca_ok=alpaca_ok,
            database_ok=database_ok,
            latency_ms=latency,
            errors=errors,
            timestamp=datetime.utcnow(),
        )
        logger.info("Health check complete", extra={"status": status})
        return report
