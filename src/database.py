import logging
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Union

from src.config import Config

logger = logging.getLogger(__name__)


class Database:
    """
    Centralized Database Manager.
    Handles connection pooling (thread-local) and schema management.
    """
    
    _local = threading.local()

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or Config.DB_PATH
        # We don't init schema automatically here to allow flexibility (live vs backtest)
        # But for backward compatibility, we might need to.
        # For now, let's assume the caller will call init_schema if needed, 
        # or we check on connection.

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a thread-local connection."""
        if not hasattr(self._local, "connections"):
            self._local.connections = {}
            
        if self.db_path not in self._local.connections:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for concurrency
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
            except sqlite3.Error as e:
                logger.warning(f"Failed to enable WAL mode: {e}")
            self._local.connections[self.db_path] = conn
            
        return self._local.connections[self.db_path]

    def close(self) -> None:
        """Closes the thread-local connection."""
        if hasattr(self._local, "connections") and self.db_path in self._local.connections:
            self._local.connections[self.db_path].close()
            del self._local.connections[self.db_path]

    def init_live_schema(self) -> None:
        """Initialize the live trading schema (Positions, Executions)."""
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
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(create_positions_table)
            cursor.execute(create_executions_table)
            conn.commit()
            logger.info(f"Live Trading Schema initialized at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Live Schema initialization failed: {e}")
            raise

    def init_backtest_schema(self) -> None:
        """Initialize the backtest analytics schema."""
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

        # 3. ML Datasets
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
            value REAL,
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
            conn = self._get_connection()
            cursor = conn.cursor()
            for stmt in statements:
                cursor.execute(stmt)
            conn.commit()
            logger.info(f"Backtest Schema initialized at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Backtest Schema initialization failed: {e}")
            raise

    # --- Live Trading Methods ---

    def create_position(self, client_oid_base: str, side: str, qty: float, entry_price: float, stop_loss: float) -> int:
        """Creates a new parent position."""
        try:
            conn = self._get_connection()
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
            conn = self._get_connection()
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
            conn = self._get_connection()
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
            conn = self._get_connection()
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
            conn = self._get_connection()
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
            conn = self._get_connection()
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

    # --- Run & Decision Logging (Live & Backtest) ---

    def register_run(self, run_id: str, run_type: str = "LIVE", notes: str = "") -> None:
        """Registers a new run (session) in the database."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO bt_runs (run_id, run_type, notes)
                VALUES (?, ?, ?)
                """,
                (run_id, run_type, notes),
            )
            conn.commit()
            logger.info(f"Registered run: {run_id} ({run_type})")
        except sqlite3.Error as e:
            logger.error(f"Failed to register run: {e}")
            raise

    def log_decision(self, run_id: str, analysis: Dict[str, Any]) -> None:
        """Logs a trading decision to the database."""
        try:
            # Extract fields from analysis dict
            # Map analysis keys to table columns
            
            # Basic Info
            timestamp = analysis.get("timestamp")
            if hasattr(timestamp, "isoformat"):
                timestamp = timestamp.isoformat()
            elif timestamp is not None:
                timestamp = str(timestamp)

            regime = analysis.get("regime", "Unknown")
            action = analysis.get("signal", "NEUTRAL") # Signal is the action (LONG/SHORT/NEUTRAL)
            side = "LONG" if action == "LONG" else ("SHORT" if action == "SHORT" else None)
            reason = analysis.get("reason", "")
            
            # ML
            ml_score = analysis.get("ml_score")
            
            # Market Data (Snapshot)
            # Try to get OHLCV from decision_context first, then top-level
            context = analysis.get("decision_context", {})
            
            open_price = context.get("open", analysis.get("open"))
            high_price = context.get("high", analysis.get("high"))
            low_price = context.get("low", analysis.get("low"))
            close_price = context.get("close", analysis.get("close"))
            volume = context.get("volume", analysis.get("volume"))
            
            params = {
                "run_id": run_id,
                "timestamp": timestamp,
                "regime": regime,
                "action": action,
                "side": side,
                "reason_string": reason,
                "ml_score": ml_score,
                
                # Market Snapshot
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                
                # Indicators
                "rsi": context.get("rsi", analysis.get("rsi")),
                "atr": context.get("atr", analysis.get("atr")),
                "adx": context.get("adx", analysis.get("adx")),
                "ema200": context.get("ema200", analysis.get("ema200")),
                "ema200_1h": context.get("ema200_1h"),
                
                # Bands
                "bb_lower": context.get("bb_lower"),
                "bb_mid": context.get("bb_mid"),
                "bb_upper": context.get("bb_upper"),
                "kc_lower": context.get("kc_lower"),
                "kc_upper": context.get("kc_upper"),
                
                # Microstructure
                "obi": context.get("obi"),
                "spread": context.get("spread"),
                "market_depth_ratio": context.get("market_depth_ratio"),
                
                # Flags (0/1)
                "is_ranging": 1 if context.get("is_ranging", analysis.get("is_ranging")) else 0,
                "is_uptrend_1h": 1 if context.get("is_uptrend_1h", analysis.get("is_uptrend")) else 0,
                "is_downtrend_1h": 1 if context.get("is_downtrend_1h", analysis.get("is_downtrend")) else 0,
                "touched_band": 1 if context.get("touched_band") else 0,
                "is_green": 1 if context.get("is_green") else 0,
                "rsi_hook": 1 if context.get("rsi_hook") else 0,
                "confirmed_1m": 1 if context.get("confirmed_1m") else 0,
                "obi_filter": 1 if context.get("obi_filter") else 0,
                "ml_filter": 1 if context.get("ml_filter") else 0,
                
            # Execution (if any) - usually updated later, but here we log the decision
            "execution_price": analysis.get("current_price", close_price), # Snapshot price
            }
        
            # Construct Query dynamically based on available keys that match columns
            # For simplicity, we'll use a fixed set of common columns
            
            query = """
                INSERT INTO bt_decisions (
                    run_id, timestamp, regime, action, side, reason_string, ml_score,
                    open, high, low, close, volume,
                    rsi, atr, adx, ema200, ema200_1h,
                    bb_lower, bb_mid, bb_upper, kc_lower, kc_upper,
                    obi, spread, market_depth_ratio,
                    is_ranging, is_uptrend_1h, is_downtrend_1h,
                    touched_band, is_green, rsi_hook, confirmed_1m, obi_filter, ml_filter,
                    execution_price
                )
                VALUES (
                    :run_id, :timestamp, :regime, :action, :side, :reason_string, :ml_score,
                    :open, :high, :low, :close, :volume,
                    :rsi, :atr, :adx, :ema200, :ema200_1h,
                    :bb_lower, :bb_mid, :bb_upper, :kc_lower, :kc_upper,
                    :obi, :spread, :market_depth_ratio,
                    :is_ranging, :is_uptrend_1h, :is_downtrend_1h,
                    :touched_band, :is_green, :rsi_hook, :confirmed_1m, :obi_filter, :ml_filter,
                    :execution_price
                )
            """
            
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            
        except sqlite3.Error as e:
            # Don't crash the bot if logging fails, just log error
            logger.error(f"Failed to log decision: {e}")

    # --- Generic Query Methods (for BacktestAnalytics) ---
    
    def execute(self, query: str, params: Union[tuple, Dict[str, Any]] = ()) -> sqlite3.Cursor:
        """Executes a query and returns the cursor."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor
        
    def commit(self) -> None:
        """Commits the current transaction."""
        conn = self._get_connection()
        conn.commit()

    def executemany(self, query: str, params_list: List[Any]) -> None:
        """Executes many params against a query."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.executemany(query, params_list)
        conn.commit()
