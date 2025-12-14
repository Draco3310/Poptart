import logging
import os
import sqlite3
import sys

import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.config import Config

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DebugTrades")


def debug_trades() -> None:
    db_path = Config.DB_PATH
    conn = sqlite3.connect(db_path)

    # List all WFO runs and trade counts
    query = """
    SELECT r.run_id, r.created_at, COUNT(t.id) as trade_count
    FROM bt_runs r
    LEFT JOIN bt_trades t ON r.run_id = t.run_id
    WHERE r.run_type = 'WFO_TEST'
    GROUP BY r.run_id
    ORDER BY r.created_at DESC
    """

    runs = pd.read_sql_query(query, conn)
    logger.info(runs)


if __name__ == "__main__":
    debug_trades()
