"""Emergency procedures for live trading."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from trading_system.core.logger import get_logger

logger = get_logger(__name__)


async def manual_emergency_flatten(exchange: str = "alpaca") -> None:
    """
    CLI entry point for manual emergency flatten.
    Usage: python -m trading_system.live.emergency
    """
    logger.critical("MANUAL EMERGENCY FLATTEN INITIATED")

    if exchange == "alpaca":
        from trading_system.config.settings import settings
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(
            settings.ALPACA_API_KEY,
            settings.ALPACA_API_SECRET,
            settings.ALPACA_BASE_URL,
        )
        api.close_all_positions()
        logger.critical("All Alpaca positions closed")

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "reason": "MANUAL_EMERGENCY_FLATTEN",
        "exchange": exchange,
    }
    halt_path = Path(__file__).parent.parent / "halt_report.json"
    halt_path.write_text(json.dumps(report, indent=2))
    print(f"Emergency flatten complete. Halt report: {halt_path}")


if __name__ == "__main__":
    asyncio.run(manual_emergency_flatten())
