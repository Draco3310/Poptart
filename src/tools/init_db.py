import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.database import Database
from src.core.backtest_analytics import BacktestAnalytics

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("InitDB")

def init_db():
    logger.info("Initializing Database...")
    
    # Initialize Trading Tables (positions, executions)
    try:
        db = Database()
        logger.info("Trading tables initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize trading tables: {e}")
        sys.exit(1)

    # Initialize Analytics Tables (bt_runs, bt_trades, etc.)
    try:
        analytics = BacktestAnalytics()
        logger.info("Analytics tables initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize analytics tables: {e}")
        sys.exit(1)

    logger.info("Database initialization complete.")

if __name__ == "__main__":
    init_db()
