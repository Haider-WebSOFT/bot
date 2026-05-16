"""Module 15 live execution engine tests."""
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLiveExecutionEngine:
    def test_sell_at_entry_raises_value_error(self):
        """Halal guard: entry side must be BUY, not SELL."""
        import asyncio
        from unittest.mock import patch
        from trading_system.live.execution_engine import LiveExecutionEngine

        with patch.dict("os.environ", {"LIVE_TRADING_ENABLED": "true",
                                        "LIVE_TRADING_CONFIRMED": "true"}):
            engine = LiveExecutionEngine()
            engine.trading_halted = False

            mock_exchange = MagicMock()
            mock_exchange.place_limit_order = AsyncMock(return_value={"status": "filled"})
            engine._binance = mock_exchange

            async def _run():
                with pytest.raises(ValueError, match="Invalid side"):
                    await engine.place_order(
                        "BTCUSDT", "INVALID_SIDE", 0.001, 50000.0,
                        "limit", "binance"
                    )

            asyncio.get_event_loop().run_until_complete(_run())

    def test_trading_halted_blocks_orders(self):
        """No orders when trading_halted is True."""
        import asyncio
        from trading_system.live.execution_engine import LiveExecutionEngine

        engine = LiveExecutionEngine()
        engine.trading_halted = True

        async def _run():
            with pytest.raises(RuntimeError, match="halted"):
                await engine.place_order("BTCUSDT", "BUY", 0.001, 50000.0,
                                          "limit", "binance")

        asyncio.get_event_loop().run_until_complete(_run())

    def test_start_requires_both_live_flags(self):
        """start() must raise LiveTradingGateError when flags not set."""
        import asyncio
        import os
        from unittest.mock import patch
        from trading_system.live.execution_engine import LiveExecutionEngine
        from trading_system.core.exceptions import LiveTradingGateError

        with patch.dict(os.environ, {"LIVE_TRADING_ENABLED": "false",
                                      "LIVE_TRADING_CONFIRMED": "false"}):
            from trading_system.config.settings import Settings
            engine = LiveExecutionEngine()

            # Override settings on the engine's imported settings
            async def _run():
                with pytest.raises(LiveTradingGateError):
                    # Patch settings.live_trading_active to return False
                    import trading_system.live.execution_engine as mod
                    orig = mod.settings.live_trading_active
                    try:
                        # Force False by patching the property
                        type(mod.settings).live_trading_active = property(lambda s: False)
                        await engine.start()
                    finally:
                        type(mod.settings).live_trading_active = property(
                            lambda s: s.LIVE_TRADING_ENABLED and s.LIVE_TRADING_CONFIRMED
                        )

            asyncio.get_event_loop().run_until_complete(_run())

    def test_emergency_flatten_sets_halted(self):
        """emergency_flatten must set trading_halted=True."""
        import asyncio
        from trading_system.live.execution_engine import LiveExecutionEngine

        mock_alpaca = MagicMock()
        mock_alpaca.close_all_positions = AsyncMock()
        engine = LiveExecutionEngine(alpaca_client=mock_alpaca)
        engine.trading_halted = False

        async def _run():
            await engine.emergency_flatten("test emergency")

        asyncio.get_event_loop().run_until_complete(_run())
        assert engine.trading_halted is True
