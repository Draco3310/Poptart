import asyncio
import logging
from typing import List

from src.config import PAIR_CONFIGS, Config
from src.core.risk_manager import RiskManager
from src.core.trading_agent import TradingAgent
from src.exchange import KrakenExchange
from src.notifier import TelegramNotifier

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Sentinel-Orchestrator")
logger.setLevel(logging.INFO)


async def monitor_balance(exchange: KrakenExchange) -> None:
    """Background task to log balance every 60 seconds."""
    try:
        while True:
            try:
                balance = await exchange.fetch_balance("USDT")
                logger.info(f"💰 BALANCE CHECK: {balance:.2f} USDT")
            except Exception as e:
                logger.error(f"Failed to fetch balance: {e}")
            await asyncio.sleep(60)
    except asyncio.CancelledError:
        pass


async def main() -> None:
    logger.info("Starting Poptart Gal Friday V2 - Multi-Coin Orchestrator...")

    # 1. Initialize Shared Components
    try:
        Config.validate()

        if not Config.KRAKEN_API_KEY or not Config.KRAKEN_API_SECRET:
            raise ValueError("Kraken API credentials missing")

        exchange = KrakenExchange(Config.KRAKEN_API_KEY, Config.KRAKEN_API_SECRET)
        await exchange.initialize()

        notifier = TelegramNotifier(Config.TELEGRAM_TOKEN or "", Config.TELEGRAM_CHAT_ID or "")
        await notifier.notify_startup()

        risk_manager = RiskManager()

        # Start Balance Monitor
        balance_task = asyncio.create_task(monitor_balance(exchange))

        logger.info("Shared components initialized.")

    except Exception as e:
        logger.critical(f"Initialization failed: {e}")
        return

    # 2. Initialize Agents
    agents: List[TradingAgent] = []
    try:
        for symbol, pair_config in PAIR_CONFIGS.items():
            logger.info(f"Creating Agent for {symbol}...")
            agent = TradingAgent(pair_config, exchange, notifier, risk_manager)
            await agent.initialize()
            agents.append(agent)

        logger.info(f"Initialized {len(agents)} agents: {[a.symbol for a in agents]}")

    except Exception as e:
        logger.critical(f"Agent initialization failed: {e}")
        await exchange.close()
        return

    # 3. Main Loop
    try:
        while True:
            logger.debug("Starting Tick Cycle...")

            for agent in agents:
                logger.debug(f"Ticking {agent.symbol}...")
                await agent.tick()

                # Stagger to respect API limits
                # Kraken Rate Limit: ~1 call per second per endpoint usually safe
                # Agent tick does multiple calls (OHLCV x2, BTC, OrderBook, Price)
                # So we need a decent gap.
                await asyncio.sleep(2)

            # Main Loop Interval
            # Wait before next cycle
            logger.debug("Cycle complete. Sleeping...")
            await asyncio.sleep(10)

    except asyncio.CancelledError:
        logger.info("Main loop cancelled.")
    except Exception as e:
        logger.error(f"Critical error in main loop: {e}", exc_info=True)
    finally:
        logger.info("Shutting down...")
        if "balance_task" in locals():
            balance_task.cancel()
            try:
                await balance_task
            except asyncio.CancelledError:
                pass

        await exchange.close()
        await notifier.notify_shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
