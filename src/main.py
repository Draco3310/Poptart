import asyncio
import logging
from datetime import datetime
from typing import List

from src.config import Config
from src.core.orchestrator import AgentOrchestrator
from src.core.risk_manager import RiskManager
from src.database import Database
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

        # Initialize Database & Run
        db = Database()
        db.init_live_schema()
        db.init_backtest_schema() # Ensure decision tables exist

        run_id = f"LIVE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_type = "PAPER" if Config.DRY_RUN else "LIVE"
        db.register_run(run_id, run_type, notes="Auto-started by Orchestrator")

        # Start Balance Monitor
        balance_task = asyncio.create_task(monitor_balance(exchange))

        logger.info(f"Shared components initialized. Run ID: {run_id}")

    except Exception as e:
        logger.critical(f"Initialization failed: {e}")
        return

    # 2. Initialize Orchestrator
    orchestrator = AgentOrchestrator(exchange, notifier, risk_manager, db=db, run_id=run_id)
    
    try:
        await orchestrator.initialize_agents()
    except Exception as e:
        logger.critical(f"Orchestrator initialization failed: {e}")
        await exchange.close()
        return

    # 3. Start Orchestrator
    try:
        await orchestrator.start()
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
        
        await orchestrator.stop()
        await exchange.close()
        await notifier.notify_shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
