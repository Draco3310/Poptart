import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)


class BacktestAnalytics:
    """
    Manages the Backtest Analytics database layer.
    Handles schema initialization, data ingestion, and querying.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or Config.DB_PATH
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize the analytics schema."""
        # 1. Data & Code Provenance
        create_data_sources = """
        CREATE TABLE IF NOT EXISTS data_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            exchange TEXT,
            symbol TEXT,
            timeframe TEXT,
            source_type TEXT,
            path_or_uri TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        );
        """

        create_feature_engine_versions = """
        CREATE TABLE IF NOT EXISTS feature_engine_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            code_version TEXT,
            schema_version TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        create_strategy_versions = """
        CREATE TABLE IF NOT EXISTS strategy_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            version_tag TEXT,
            code_version TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        create_config_snapshots = """
        CREATE TABLE IF NOT EXISTS config_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hash TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        # 2. Runs & Results
        create_bt_runs = """
        CREATE TABLE IF NOT EXISTS bt_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            run_type TEXT NOT NULL, -- BACKTEST, PAPER, LIVE
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            start_date DATE,
            end_date DATE,
            strategy_version_id INTEGER,
            feature_engine_version_id INTEGER,
            data_source_id INTEGER,
            config_snapshot_id INTEGER,
            code_version TEXT,
            report_path TEXT,
            debug_log_path TEXT,
            notes TEXT,
            FOREIGN KEY (strategy_version_id) REFERENCES strategy_versions (id),
            FOREIGN KEY (feature_engine_version_id) REFERENCES feature_engine_versions (id),
            FOREIGN KEY (data_source_id) REFERENCES data_sources (id),
            FOREIGN KEY (config_snapshot_id) REFERENCES config_snapshots (id)
        );
        """
        idx_bt_runs_type = (
            "CREATE INDEX IF NOT EXISTS idx_bt_runs_run_type_created_at ON bt_runs (run_type, created_at);"
        )

        create_bt_decisions = """
        CREATE TABLE IF NOT EXISTS bt_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            timestamp TIMESTAMP,
            regime TEXT,
            action TEXT,
            side TEXT,
            reason_string TEXT,
            trade_id TEXT,
            ml_score REAL,
            pre_ml_signal TEXT,
            final_signal TEXT,

            -- Execution Details
            execution_price REAL,
            execution_qty REAL,

            -- Risk Management Telemetry
            risk_reason TEXT,
            risk_qty REAL,
            risk_circuit_breaker INTEGER,
            risk_trade_gated INTEGER,
            risk_base_qty REAL,
            risk_max_vol_qty REAL,
            risk_max_spot_qty REAL,

            -- Market Snapshot
            open REAL, high REAL, low REAL, close REAL, volume REAL, volume_ma REAL,

            -- Indicators
            rsi REAL,
            bb_lower REAL, bb_mid REAL, bb_upper REAL,
            kc_lower REAL, kc_mid REAL, kc_upper REAL,
            ema200 REAL, ema200_1h REAL,
            atr REAL, adx REAL,
            obi REAL, spread REAL, market_depth_ratio REAL,

            -- Flags (0/1)
            is_ranging INTEGER,
            is_uptrend_1h INTEGER,
            is_downtrend_1h INTEGER,
            touched_band INTEGER,
            is_green INTEGER,
            rsi_hook INTEGER,
            confirmed_1m INTEGER,
            obi_filter INTEGER,
            ml_filter INTEGER,

            -- Extensibility
            extra_features TEXT,
            feature_schema_version TEXT,

            FOREIGN KEY (run_id) REFERENCES bt_runs (run_id)
        );
        """
        idx_bt_decisions_ts = (
            "CREATE INDEX IF NOT EXISTS idx_bt_decisions_run_id_timestamp ON bt_decisions (run_id, timestamp);"
        )
        idx_bt_decisions_action = (
            "CREATE INDEX IF NOT EXISTS idx_bt_decisions_run_id_regime_action_side "
            "ON bt_decisions (run_id, regime, action, side);"
        )
        idx_bt_decisions_trade = (
            "CREATE INDEX IF NOT EXISTS idx_bt_decisions_run_id_trade_id ON bt_decisions (run_id, trade_id);"
        )

        create_bt_trades = """
        CREATE TABLE IF NOT EXISTS bt_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            client_oid TEXT,
            timestamp TIMESTAMP,
            side TEXT,
            price REAL,
            amount REAL,
            fee REAL,
            pnl REAL,
            pnl_pct REAL,
            status TEXT,
            order_type TEXT,
            extra TEXT,
            FOREIGN KEY (run_id) REFERENCES bt_runs (run_id)
        );
        """
        idx_bt_trades_ts = "CREATE INDEX IF NOT EXISTS idx_bt_trades_run_id_timestamp ON bt_trades (run_id, timestamp);"
        idx_bt_trades_oid = (
            "CREATE INDEX IF NOT EXISTS idx_bt_trades_run_id_client_oid ON bt_trades (run_id, client_oid);"
        )

        create_bt_metrics = """
        CREATE TABLE IF NOT EXISTS bt_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            segment TEXT,
            metric_name TEXT NOT NULL,
            metric_value REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES bt_runs (run_id)
        );
        """
        idx_bt_metrics = (
            "CREATE INDEX IF NOT EXISTS idx_bt_metrics_run_id_segment_metric_name "
            "ON bt_metrics (run_id, segment, metric_name);"
        )

        # 3. ML Datasets (Conceptual - creating tables now for future use)
        create_ml_datasets = """
        CREATE TABLE IF NOT EXISTS ml_datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source_runs TEXT,
            feature_schema_version TEXT,
            label_definition TEXT
        );
        """

        create_ml_labels = """
        CREATE TABLE IF NOT EXISTS ml_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            label_type TEXT,
            value REAL, -- Can store numeric label here. For text labels, might need another col or cast.
            metadata TEXT,
            FOREIGN KEY (dataset_id) REFERENCES ml_datasets (id)
        );
        """

        create_ml_samples = """
        CREATE TABLE IF NOT EXISTS ml_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            run_id TEXT,
            decision_id INTEGER,
            trade_id INTEGER,
            timestamp TIMESTAMP,
            label_id INTEGER,
            feature_ref TEXT,
            FOREIGN KEY (dataset_id) REFERENCES ml_datasets (id),
            FOREIGN KEY (run_id) REFERENCES bt_runs (run_id),
            FOREIGN KEY (decision_id) REFERENCES bt_decisions (id),
            FOREIGN KEY (trade_id) REFERENCES bt_trades (id),
            FOREIGN KEY (label_id) REFERENCES ml_labels (id)
        );
        """

        statements = [
            create_data_sources,
            create_feature_engine_versions,
            create_strategy_versions,
            create_config_snapshots,
            create_bt_runs,
            idx_bt_runs_type,
            create_bt_decisions,
            idx_bt_decisions_ts,
            idx_bt_decisions_action,
            idx_bt_decisions_trade,
            create_bt_trades,
            idx_bt_trades_ts,
            idx_bt_trades_oid,
            create_bt_metrics,
            idx_bt_metrics,
            create_ml_datasets,
            create_ml_labels,
            create_ml_samples,
        ]

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                for stmt in statements:
                    cursor.execute(stmt)
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Analytics DB initialization failed: {e}")
            raise

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
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, run_data)
            conn.commit()
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

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Check existence
            cursor.execute("SELECT id FROM config_snapshots WHERE hash = ?", (config_hash,))
            row = cursor.fetchone()
            if row:
                return int(row["id"])

            # Insert
            cursor.execute("INSERT INTO config_snapshots (hash, content) VALUES (?, ?)", (config_hash, content))
            conn.commit()
            if cursor.lastrowid is None:
                raise ValueError("Failed to retrieve last row ID")
            return int(cursor.lastrowid)

    def get_or_create_strategy_version(self, name: str, version_tag: str, code_version: str = "") -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM strategy_versions WHERE name = ? AND version_tag = ? AND code_version = ?",
                (name, version_tag, code_version),
            )
            row = cursor.fetchone()
            if row:
                return int(row["id"])

            cursor.execute(
                "INSERT INTO strategy_versions (name, version_tag, code_version) VALUES (?, ?, ?)",
                (name, version_tag, code_version),
            )
            conn.commit()
            if cursor.lastrowid is None:
                raise ValueError("Failed to retrieve last row ID")
            return int(cursor.lastrowid)

    def bulk_insert_decisions(self, decisions: List[Dict[str, Any]]) -> None:
        if not decisions:
            return

        # Define valid columns for bt_decisions
        valid_columns = {
            "run_id",
            "timestamp",
            "regime",
            "action",
            "side",
            "reason_string",
            "trade_id",
            "ml_score",
            "pre_ml_signal",
            "final_signal",
            "execution_price",
            "execution_qty",
            "risk_reason",
            "risk_qty",
            "risk_circuit_breaker",
            "risk_trade_gated",
            "risk_base_qty",
            "risk_max_vol_qty",
            "risk_max_spot_qty",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "volume_ma",
            "rsi",
            "bb_lower",
            "bb_mid",
            "bb_upper",
            "kc_lower",
            "kc_mid",
            "kc_upper",
            "ema200",
            "ema200_1h",
            "atr",
            "adx",
            "obi",
            "spread",
            "market_depth_ratio",
            "is_ranging",
            "is_uptrend_1h",
            "is_downtrend_1h",
            "touched_band",
            "is_green",
            "rsi_hook",
            "confirmed_1m",
            "obi_filter",
            "ml_filter",
            "extra_features",
            "feature_schema_version",
        }

        clean_decisions = []
        for d in decisions:
            clean_d = {}

            # Flatten trade_info if present
            trade_info = d.get("trade_info") or {}
            if trade_info:
                clean_d["trade_id"] = trade_info.get("trade_id")
                clean_d["execution_price"] = trade_info.get("execution_price")
                clean_d["execution_qty"] = trade_info.get("execution_qty")
                # Map exit_reason to reason_string if not already set
                if not d.get("reason_string") and trade_info.get("exit_reason"):
                    clean_d["reason_string"] = trade_info.get("exit_reason")

            # Copy other fields
            for k, v in d.items():
                if k == "trade_info":
                    continue

                if k == "timestamp" and isinstance(v, (pd.Timestamp, datetime)):
                    clean_d[k] = str(v)
                elif k in valid_columns:
                    clean_d[k] = v

            # Ensure all valid columns exist (fill with None)
            for col in valid_columns:
                if col not in clean_d:
                    clean_d[col] = None

            clean_decisions.append(clean_d)

        if not clean_decisions:
            return

        columns = list(valid_columns)
        placeholders = ", ".join([":" + col for col in columns])
        col_names = ", ".join(columns)

        query = f"INSERT INTO bt_decisions ({col_names}) VALUES ({placeholders})"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, clean_decisions)
            conn.commit()

    def bulk_insert_trades(self, trades: List[Dict[str, Any]]) -> None:
        if not trades:
            return

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

        # Prepare list of dicts with only valid columns and converted timestamps
        clean_trades = []
        for t in trades:
            clean_t = {}
            for col in valid_columns:
                val = t.get(col)
                if col == "timestamp" and isinstance(val, (pd.Timestamp, datetime)):
                    val = str(val)
                if col == "extra" and isinstance(val, (dict, list)):
                    val = json.dumps(val)
                clean_t[col] = val
            clean_trades.append(clean_t)

        if not clean_trades:
            return

        columns = list(valid_columns)
        placeholders = ", ".join([":" + col for col in columns])
        col_names = ", ".join(columns)

        query = f"INSERT INTO bt_trades ({col_names}) VALUES ({placeholders})"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, clean_trades)
            conn.commit()

    def insert_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        if not metrics:
            return

        query = (
            "INSERT INTO bt_metrics (run_id, segment, metric_name, metric_value) "
            "VALUES (:run_id, :segment, :metric_name, :metric_value)"
        )

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, metrics)
            conn.commit()

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
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(limit,))

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        query = """
        SELECT r.*, s.name as strategy_name, s.version_tag, c.content as config_json
        FROM bt_runs r
        LEFT JOIN strategy_versions s ON r.strategy_version_id = s.id
        LEFT JOIN config_snapshots c ON r.config_snapshot_id = c.id
        WHERE r.run_id = ?
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (run_id,))
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

        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_trades(self, run_id: str) -> pd.DataFrame:
        query = "SELECT * FROM bt_trades WHERE run_id = ?"
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(run_id,))

    def get_metrics(self, run_id: str) -> pd.DataFrame:
        query = "SELECT * FROM bt_metrics WHERE run_id = ?"
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(run_id,))

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

        with self._get_connection() as conn:
            cursor = conn.cursor()
            v1 = cursor.execute(v1_query, (run_id,)).fetchone()["count"]
            v2 = cursor.execute(v2_query, (run_id,)).fetchone()["count"]

        return {"not_ranging_entries": v1, "no_band_touch_entries": v2}
