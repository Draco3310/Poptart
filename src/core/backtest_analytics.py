import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import Config
from src.database import Database

logger = logging.getLogger(__name__)


class BacktestAnalytics:
    """
    Manages the Backtest Analytics database layer.
    Handles schema initialization, data ingestion, and querying.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db = Database(db_path or Config.DB_PATH)
        self.db.init_backtest_schema()
        self._decision_columns = self._get_table_columns("bt_decisions")

    def _get_table_columns(self, table_name: str) -> set:
        cursor = self.db.execute(f"PRAGMA table_info({table_name})")
        return {row["name"] for row in cursor.fetchall()}

    # --- Ingestion Helpers ---

    def register_run(self, run_data: Dict[str, Any]) -> int:
        """
        Registers a new backtest run.
        Returns the inserted row ID.
        """
        query = """
        INSERT INTO bt_runs (
            run_id, run_type, start_date, end_date,
            strategy_version_id, feature_engine_version_id, data_source_id, config_snapshot_id,
            code_version, report_path, debug_log_path, notes
        ) VALUES (
            :run_id, :run_type, :start_date, :end_date,
            :strategy_version_id, :feature_engine_version_id, :data_source_id, :config_snapshot_id,
            :code_version, :report_path, :debug_log_path, :notes
        )
        """
        cursor = self.db.execute(query, run_data)
        self.db.commit()
        if cursor.lastrowid is None:
            raise ValueError("Failed to retrieve last row ID")
        return int(cursor.lastrowid)

    def get_or_create_config_snapshot(self, config_dict: Dict[str, Any]) -> int:
        """
        Stores config snapshot if not exists. Returns ID.
        """
        # Sort keys for consistent hashing
        content = json.dumps(config_dict, sort_keys=True)
        # Simple hash
        import hashlib

        config_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        cursor = self.db.execute("SELECT id FROM config_snapshots WHERE hash = ?", (config_hash,))
        row = cursor.fetchone()
        if row:
            return int(row["id"])

        # Insert
        cursor = self.db.execute("INSERT INTO config_snapshots (hash, content) VALUES (?, ?)", (config_hash, content))
        self.db.commit()
        if cursor.lastrowid is None:
            raise ValueError("Failed to retrieve last row ID")
        return int(cursor.lastrowid)

    def get_or_create_strategy_version(self, name: str, version_tag: str, code_version: str = "") -> int:
        cursor = self.db.execute(
            "SELECT id FROM strategy_versions WHERE name = ? AND version_tag = ? AND code_version = ?",
            (name, version_tag, code_version),
        )
        row = cursor.fetchone()
        if row:
            return int(row["id"])

        cursor = self.db.execute(
            "INSERT INTO strategy_versions (name, version_tag, code_version) VALUES (?, ?, ?)",
            (name, version_tag, code_version),
        )
        self.db.commit()
        if cursor.lastrowid is None:
            raise ValueError("Failed to retrieve last row ID")
        return int(cursor.lastrowid)

    def bulk_insert_decisions(self, decisions: List[Dict[str, Any]]) -> None:
        if not decisions:
            return

        # Use Pandas for efficient processing and flattening
        # json_normalize flattens nested dicts (e.g. trade_info.trade_id)
        df = pd.json_normalize(decisions)

        # Rename flattened trade_info columns
        rename_map = {
            "trade_info.trade_id": "trade_id",
            "trade_info.execution_price": "execution_price",
            "trade_info.execution_qty": "execution_qty",
            "trade_info.exit_reason": "exit_reason",
        }
        df.rename(columns=rename_map, inplace=True)

        # Map exit_reason to reason_string if needed
        if "exit_reason" in df.columns:
            if "reason_string" not in df.columns:
                df["reason_string"] = None
            df["reason_string"] = df["reason_string"].fillna(df["exit_reason"])

        # Filter for valid columns only (using cached schema)
        valid_cols = list(self._decision_columns.intersection(df.columns))
        df = df[valid_cols]

        # Convert timestamps to string (SQLite requirement)
        if "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"].astype(str)

        # Convert to list of dicts
        clean_decisions = df.to_dict("records")

        if not clean_decisions:
            return

        placeholders = ", ".join([":" + col for col in valid_cols])
        col_names = ", ".join(valid_cols)
        query = f"INSERT INTO bt_decisions ({col_names}) VALUES ({placeholders})"

        self.db.executemany(query, clean_decisions)

    def bulk_insert_trades(self, trades: List[Dict[str, Any]]) -> None:
        if not trades:
            return

        # Use Pandas for efficient processing
        df = pd.DataFrame(trades)

        # Filter keys to match schema
        valid_columns = {
            "run_id",
            "client_oid",
            "timestamp",
            "side",
            "price",
            "amount",
            "fee",
            "pnl",
            "pnl_pct",
            "status",
            "order_type",
            "extra",
        }
        
        # Ensure all valid columns exist
        for col in valid_columns:
            if col not in df.columns:
                df[col] = None

        # Filter columns
        df = df[list(valid_columns)]

        # Convert timestamps
        if "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"].astype(str)

        # Serialize 'extra' column if it exists and has content
        if "extra" in df.columns:
            # Vectorized serialization is tricky, but apply is cleaner than loop
            df["extra"] = df["extra"].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

        # Convert to list of dicts
        clean_trades = df.to_dict("records")

        if not clean_trades:
            return

        columns = list(valid_columns)
        placeholders = ", ".join([":" + col for col in columns])
        col_names = ", ".join(columns)

        query = f"INSERT INTO bt_trades ({col_names}) VALUES ({placeholders})"

        self.db.executemany(query, clean_trades)

    def insert_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        if not metrics:
            return

        query = (
            "INSERT INTO bt_metrics (run_id, segment, metric_name, metric_value) "
            "VALUES (:run_id, :segment, :metric_name, :metric_value)"
        )

        self.db.executemany(query, metrics)

    # --- Query Helpers ---

    def list_runs(self, limit: int = 20) -> pd.DataFrame:
        query = """
        SELECT r.run_id, r.run_type, r.created_at, r.start_date, r.end_date,
               s.name as strategy, s.version_tag,
               (SELECT COUNT(*) FROM bt_trades t WHERE t.run_id = r.run_id) as trade_count,
               (SELECT metric_value FROM bt_metrics m
                WHERE m.run_id = r.run_id AND m.metric_name = 'pnl' AND m.segment = 'overall') as total_pnl
        FROM bt_runs r
        LEFT JOIN strategy_versions s ON r.strategy_version_id = s.id
        ORDER BY r.created_at DESC
        LIMIT ?
        """
        # pd.read_sql_query needs a connection object, not a cursor
        # We can access the underlying connection from Database
        return pd.read_sql_query(query, self.db._get_connection(), params=(limit,))

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        query = """
        SELECT r.*, s.name as strategy_name, s.version_tag, c.content as config_json
        FROM bt_runs r
        LEFT JOIN strategy_versions s ON r.strategy_version_id = s.id
        LEFT JOIN config_snapshots c ON r.config_snapshot_id = c.id
        WHERE r.run_id = ?
        """
        cursor = self.db.execute(query, (run_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_decisions(
        self, run_id: str, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None
    ) -> pd.DataFrame:
        query = "SELECT * FROM bt_decisions WHERE run_id = ?"
        params: List[Any] = [run_id]

        if filters:
            for k, v in filters.items():
                query += f" AND {k} = ?"
                params.append(v)

        if limit:
            query += f" LIMIT {limit}"

        return pd.read_sql_query(query, self.db._get_connection(), params=params)

    def get_trades(self, run_id: str) -> pd.DataFrame:
        query = "SELECT * FROM bt_trades WHERE run_id = ?"
        return pd.read_sql_query(query, self.db._get_connection(), params=(run_id,))

    def get_metrics(self, run_id: str) -> pd.DataFrame:
        query = "SELECT * FROM bt_metrics WHERE run_id = ?"
        return pd.read_sql_query(query, self.db._get_connection(), params=(run_id,))

    def get_invariant_violations(self, run_id: str) -> Dict[str, Any]:
        """
        Checks for common invariant violations in Mean Reversion entries.
        Returns a summary dictionary.
        """
        # This is a simplified check. For full details, one would query the raw data.
        # Here we use SQL to aggregate.

        # Violation 1: ENTRY_LONG in CHOP but is_ranging = 0
        v1_query = """
        SELECT COUNT(*) as count
        FROM bt_decisions
        WHERE run_id = ? AND action = 'ENTRY_LONG' AND regime = 'CHOP' AND is_ranging = 0
        """

        # Violation 2: ENTRY_LONG but not touched_band
        v2_query = """
        SELECT COUNT(*) as count
        FROM bt_decisions
        WHERE run_id = ? AND action = 'ENTRY_LONG' AND regime = 'CHOP' AND touched_band = 0
        """

        # Violation 3: ENTRY_LONG but RSI >= 30 (assuming 30 is hardcoded check, ideally passed in)
        # We'll just return the distribution of RSI for entries

        cursor = self.db.execute(v1_query, (run_id,))
        v1 = cursor.fetchone()["count"]
        
        cursor = self.db.execute(v2_query, (run_id,))
        v2 = cursor.fetchone()["count"]

        return {"not_ranging_entries": v1, "no_band_touch_entries": v2}
