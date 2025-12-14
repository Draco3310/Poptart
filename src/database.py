import logging
import sqlite3
from typing import Optional

from src.config import Config

logger = logging.getLogger(__name__)


class Database:
    def __init__(self) -> None:
        self.db_path = Config.DB_PATH
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize the database schema (Parent-Child)."""
        create_positions_table = """
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_oid_base TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL, -- OPEN, PARTIAL, CLOSED
            side TEXT NOT NULL, -- LONG, SHORT
            remaining_qty REAL NOT NULL,
            entry_price REAL NOT NULL,
            stop_loss REAL NOT NULL,
            tp1_hit BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        create_executions_table = """
        CREATE TABLE IF NOT EXISTS executions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            position_id INTEGER NOT NULL,
            order_id TEXT, -- Kraken Order ID
            side TEXT NOT NULL, -- BUY, SELL
            qty REAL NOT NULL,
            price REAL NOT NULL,
            reason TEXT NOT NULL, -- ENTRY, TP1, STOP_LOSS, TRAILING_STOP
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (position_id) REFERENCES positions (id)
        );
        """

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(create_positions_table)
                cursor.execute(create_executions_table)
                conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def create_position(self, client_oid_base: str, side: str, qty: float, entry_price: float, stop_loss: float) -> int:
        """Creates a new parent position."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO positions (
                        client_oid_base, status, side, remaining_qty, entry_price, stop_loss, tp1_hit
                    )
                    VALUES (?, 'OPEN', ?, ?, ?, ?, 0)
                """,
                    (client_oid_base, side, qty, entry_price, stop_loss),
                )
                conn.commit()
                if cursor.lastrowid is None:
                    raise ValueError("Failed to retrieve last row ID")
                return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Failed to create position: {e}")
            raise

    def add_execution(self, position_id: int, order_id: str, side: str, qty: float, price: float, reason: str) -> None:
        """Records a specific fill (execution)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO executions (position_id, order_id, side, qty, price, reason)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (position_id, order_id, side, qty, price, reason),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to add execution: {e}")
            raise

    def update_position_status(self, position_id: int, status: str, remaining_qty: float) -> None:
        """Updates the position status and remaining quantity."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE positions
                    SET status = ?, remaining_qty = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """,
                    (status, remaining_qty, position_id),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to update position status: {e}")
            raise

    def update_stop_loss(self, position_id: int, new_sl: float) -> None:
        """Updates the stop loss price."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE positions
                    SET stop_loss = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """,
                    (new_sl, position_id),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to update stop loss: {e}")
            raise

    def set_tp1_hit(self, position_id: int) -> None:
        """Marks TP1 as hit."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE positions
                    SET tp1_hit = 1, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """,
                    (position_id,),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to set TP1 hit: {e}")
            raise

    def get_open_position(self) -> Optional[dict]:
        """Retrieves the current open position (if any). Assumes only one open position at a time."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM positions
                    WHERE status IN ('OPEN', 'PARTIAL')
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
        except sqlite3.Error as e:
            logger.error(f"Failed to get open position: {e}")
            raise
