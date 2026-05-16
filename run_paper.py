"""Paper trading runner — start this to begin paper trading."""
from __future__ import annotations

import asyncio
import signal
import sys
from datetime import datetime

from trading_system.agents.early_momentum_agent import EarlyMomentumAgent
from trading_system.agents.ensemble import EnsembleEngine
from trading_system.agents.execution_quality_agent import ExecutionQualityAgent
from trading_system.agents.liquidity_agent import LiquidityAgent
from trading_system.agents.momentum_agent import MomentumAgent
from trading_system.agents.regime_detector import RegimeDetector
from trading_system.agents.sentiment_agent import SentimentAgent
from trading_system.agents.trend_agent import TrendAgent
from trading_system.agents.volatility_agent import VolatilityAgent
from trading_system.agents.volume_agent import VolumeAgent
from trading_system.clients.binance_client import BinanceClient
from trading_system.clients.alpaca_client import AlpacaClient
from trading_system.config.constants import CRYPTO_UNIVERSE
from trading_system.scanner.universe import UniverseManager
from trading_system.config.settings import settings
from trading_system.core.logger import get_logger
from trading_system.data.feed import DataFeed
from trading_system.paper_trading.paper_engine import PaperTradingEngine
from trading_system.paper_trading.trade_journal import TradeJournal
from trading_system.risk.portfolio_risk import PortfolioRiskEngine as PortfolioRiskManager
from trading_system.risk.position_sizer import PositionSizer
from trading_system.risk.trade_manager import TradeManager
from trading_system.scanner.crypto_scanner import CryptoScanner
from trading_system.scanner.stock_scanner import StockScanner

logger = get_logger(__name__)

SCAN_INTERVAL = 60 * settings.SCAN_INTERVAL_MINUTES  # seconds


def _build_agents() -> dict:
    return {
        "TrendAgent": TrendAgent(),
        "MomentumAgent": MomentumAgent(),
        "VolumeAgent": VolumeAgent(),
        "VolatilityAgent": VolatilityAgent(),
        "LiquidityAgent": LiquidityAgent(),
        "EarlyMomentumAgent": EarlyMomentumAgent(),
        "SentimentAgent": SentimentAgent(),
    }


async def run() -> None:
    print("=" * 60)
    print("  HALAL PAPER TRADING ENGINE  —  Spot Long Only")
    print(f"  Capital: $1000.00")
    print(f"  Scan interval: {settings.SCAN_INTERVAL_MINUTES} min")
    print("=" * 60)

    # Clients
    binance = BinanceClient()
    alpaca = AlpacaClient()

    # Core components
    data_feed = DataFeed(binance, alpaca)
    regime_detector = RegimeDetector()
    ensemble = EnsembleEngine()
    agents = _build_agents()
    sizer = PositionSizer()

    # Risk
    portfolio_risk = PortfolioRiskManager()
    trade_manager = TradeManager()
    execution_agent = ExecutionQualityAgent()

    # Journal + engine
    initial_capital = float(
        __import__("os").environ.get("INITIAL_CAPITAL", "1000.0")
    )
    journal = TradeJournal()
    engine = PaperTradingEngine(
        initial_capital=initial_capital,
        journal=journal,
        portfolio_risk=portfolio_risk,
        trade_manager=trade_manager,
        execution_agent=execution_agent,
    )

    crypto_scanner = CryptoScanner()
    stock_scanner = StockScanner()

    scan_count = 0
    print("\nStarting scan loop — Ctrl+C to stop\n")

    while True:
        scan_count += 1
        now = datetime.utcnow().strftime("%H:%M:%S UTC")
        print(f"[{now}] Scan #{scan_count} ...", end="  ", flush=True)

        try:
            # Fetch all snapshots first for position updates
            universe = UniverseManager()
            stock_symbols = await universe.get_tradeable_stocks()

            crypto_snaps = await data_feed.get_all_snapshots(CRYPTO_UNIVERSE, "crypto")
            stock_snaps = await data_feed.get_all_snapshots(stock_symbols, "stock")
            all_snaps = crypto_snaps + stock_snaps

            # Update open positions (stop loss / take profit checks)
            await engine.update_all_positions(all_snaps)

            # Scan for new opportunities
            crypto_cards = await crypto_scanner.scan(
                data_feed, regime_detector, ensemble, agents
            )
            stock_cards = await stock_scanner.scan(
                data_feed, alpaca, regime_detector, ensemble, agents
            )
            all_cards = crypto_cards + stock_cards

            opened = 0
            snap_map = {s.symbol: s for s in all_snaps}
            for card in all_cards:
                snap = snap_map.get(card.symbol)
                if not snap:
                    continue
                try:
                    sizing = sizer.calculate(
                        snap,
                        regime_detector.detect(snap),
                        account_equity=engine.equity,
                        strategy_type=card.strategy_type,
                    )
                except Exception:
                    continue
                if await engine.process_opportunity(card, snap, sizing):
                    opened += 1

            open_pos = len(engine._open_positions)
            pnl = engine.equity - engine.initial_capital
            print(f"opportunities={len(all_cards)}  opened={opened}  "
                  f"positions={open_pos}  equity=${engine.equity:,.2f}  "
                  f"PnL={pnl:+.2f}")

        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"ERROR: {exc}")
            logger.exception("Scan loop error", extra={"error": str(exc)})

        await asyncio.sleep(SCAN_INTERVAL)


def main() -> None:
    def _stop(sig, frame):
        print("\n\nStopping paper trading engine...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    asyncio.run(run())


if __name__ == "__main__":
    main()
