import logging
import sqlite3

from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def delete_run(run_id: str) -> None:
    db_path = Config.DB_PATH
    logger.info(f"Deleting run {run_id} from {db_path}...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Delete dependent tables first
        cursor.execute("DELETE FROM bt_decisions WHERE run_id = ?", (run_id,))
        logger.info(f"Deleted decisions for {run_id}")

        cursor.execute("DELETE FROM bt_trades WHERE run_id = ?", (run_id,))
        logger.info(f"Deleted trades for {run_id}")

        cursor.execute("DELETE FROM bt_metrics WHERE run_id = ?", (run_id,))
        logger.info(f"Deleted metrics for {run_id}")

        # Delete run
        cursor.execute("DELETE FROM bt_runs WHERE run_id = ?", (run_id,))
        logger.info(f"Deleted run record for {run_id}")

        conn.commit()
        logger.info("Deletion complete.")

    except Exception as e:
        logger.error(f"Error deleting run: {e}")
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    delete_run("20251202_173113")
