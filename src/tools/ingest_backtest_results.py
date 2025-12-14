import argparse
import glob
import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.getcwd())

from src.config import Config
from src.core.backtest_analytics import BacktestAnalytics

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_report_file(report_path: str) -> Dict[str, Any]:
    """Parses the backtest report text file for metadata."""
    meta = {}
    try:
        with open(report_path, "r") as f:
            content = f.read()

        # Extract Strategy
        m_strat = re.search(r"Strategy:\s+(.+)", content)
        if m_strat:
            meta["strategy_name"] = m_strat.group(1).strip()

        # Extract Dates
        # Look for "Start Date: YYYY-MM-DD" and "End Date: YYYY-MM-DD"
        m_start = re.search(r"Start Date:\s+(\d{4}-\d{2}-\d{2})", content)
        if m_start:
            meta["start_date"] = m_start.group(1)

        m_end = re.search(r"End Date:\s+(\d{4}-\d{2}-\d{2})", content)
        if m_end:
            meta["end_date"] = m_end.group(1)

        # Extract Metrics
        # Look for "PnL:             -77.23 USDT"
        m_pnl = re.search(r"PnL:\s+([-\d\.]+)\s+USDT", content)
        if m_pnl:
            meta["total_pnl"] = float(m_pnl.group(1))

    except Exception as e:
        logger.warning(f"Failed to parse report file {report_path}: {e}")

    return meta


def normalize_decisions(df: pd.DataFrame, run_id: str) -> List[Dict[str, Any]]:
    """
    Normalizes decision log dataframe to match DB schema.
    Separates core columns from extra features.
    """
    # Core columns in DB
    core_cols = {
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
    }

    records = []

    # Convert boolean columns to int (0/1)
    # Explicitly list boolean columns to avoid clobbering string columns like 'action'
    target_bool_cols = {
        "is_ranging",
        "is_uptrend_1h",
        "is_downtrend_1h",
        "touched_band",
        "is_green",
        "rsi_hook",
        "confirmed_1m",
        "obi_filter",
        "ml_filter",
        "risk_circuit_breaker",
        "risk_trade_gated",
    }

    for col in df.columns:
        if col in target_bool_cols and col in core_cols:
            # If it's boolean type, astype(int) works. If string, map.
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
            else:
                # Handle string 'True'/'False' if present
                # We use a map that preserves existing 0/1 if they are already numeric but dtype is object
                # But safer to just map True/False strings and fillna with original or 0
                # Actually, if it's object, it might contain 'True', 'False', or already 0/1
                # Let's try to convert to numeric first
                df[col] = df[col].replace({"True": 1, "False": 0, True: 1, False: 0})
                # Force numeric, coercing errors to NaN (then fill 0)
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Replace NaN with None for SQL
    df = df.replace({np.nan: None})

    for _, row in df.iterrows():
        record = {"run_id": run_id}
        extra = {}

        for k, val in row.items():
            col_str = str(k)
            if col_str in core_cols:
                record[col_str] = val
            else:
                # Check if it's a feature (e.g. starts with feature_ or just unknown)
                # We'll put everything else in extra
                extra[col_str] = val

        if extra:
            record["extra_features"] = json.dumps(extra)

        records.append(record)

    return records


def normalize_trades(df: pd.DataFrame, run_id: str) -> List[Dict[str, Any]]:
    """
    Normalizes trades dataframe and computes PnL using FIFO.
    """
    records = []
    df = df.replace({np.nan: None})

    # Sort by timestamp to ensure FIFO order
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

    # FIFO Position Tracking
    inventory = []  # List of {'price': float, 'qty': float, 'fee': float}

    # DB columns
    db_cols = {"client_oid", "timestamp", "side", "price", "amount", "fee", "pnl", "pnl_pct", "status", "order_type"}

    for _, row in df.iterrows():
        record: Dict[str, Any] = {"run_id": run_id}
        extra = {}

        # Copy existing fields
        for k, val in row.items():
            col_str = str(k)
            if col_str in db_cols:
                # Convert Timestamp to string for SQLite
                if col_str == "timestamp" and isinstance(val, (pd.Timestamp, datetime)):
                    record[col_str] = str(val)
                else:
                    record[col_str] = val
            else:
                extra[col_str] = val

        # Compute PnL if missing
        side = str(row.get("side", "")).upper()
        price = float(row.get("price", 0.0))
        qty = float(row.get("amount", 0.0))
        fee = float(row.get("fee", 0.0))

        if side == "BUY":
            # Add to inventory
            inventory.append({"price": price, "qty": qty, "fee": fee})
            record["pnl"] = 0.0
            record["pnl_pct"] = 0.0

        elif side == "SELL":
            # FIFO Exit
            realized_pnl = 0.0
            cost_basis = 0.0
            qty_to_close = qty

            while qty_to_close > 0.00000001 and inventory:
                batch = inventory[0]

                if batch["qty"] > qty_to_close:
                    # Partial close of this batch
                    portion = qty_to_close / batch["qty"]
                    batch_cost = batch["price"] * qty_to_close
                    batch_fee = batch["fee"] * portion

                    cost_basis += batch_cost + batch_fee

                    # Update batch
                    batch["qty"] -= qty_to_close
                    batch["fee"] -= batch_fee
                    qty_to_close = 0
                else:
                    # Full close of this batch
                    batch_cost = batch["price"] * batch["qty"]
                    batch_fee = batch["fee"]

                    cost_basis += batch_cost + batch_fee

                    qty_to_close -= batch["qty"]
                    inventory.pop(0)

            proceeds = (price * qty) - fee
            realized_pnl = proceeds - cost_basis
            pnl_pct = (realized_pnl / cost_basis) * 100 if cost_basis > 0 else 0.0

            record["pnl"] = realized_pnl
            record["pnl_pct"] = pnl_pct

        if extra:
            record["extra"] = json.dumps(extra)

        records.append(record)

    return records


def ingest_run(
    run_id: str,
    report_path: str,
    debug_log_path: Optional[str],
    trades_csv: str,
    decisions_csv: str,
    analytics: BacktestAnalytics,
) -> None:
    logger.info(f"Ingesting run {run_id}...")

    # 1. Parse Metadata
    meta = parse_report_file(report_path)

    # 2. Store Config Snapshot
    config_dict = {}
    for k in dir(Config):
        if k.startswith("__"):
            continue
        v = getattr(Config, k)
        if not callable(v) and not isinstance(v, (classmethod, staticmethod)):
            config_dict[k] = v

    config_id = analytics.get_or_create_config_snapshot(config_dict)

    # 3. Store Strategy Version
    strat_name = meta.get("strategy_name", "Unknown")
    # We don't have version tag in report usually, assume v1 or parse from filename if possible
    strat_id = analytics.get_or_create_strategy_version(strat_name, "v1")

    # 4. Register Run
    run_data = {
        "run_id": run_id,
        "run_type": "BACKTEST",
        "start_date": meta.get("start_date"),
        "end_date": meta.get("end_date"),
        "strategy_version_id": strat_id,
        "feature_engine_version_id": None,  # TODO: Version FeatureEngine
        "data_source_id": None,  # TODO: Track data source
        "config_snapshot_id": config_id,
        "code_version": None,  # TODO: Get git hash
        "report_path": report_path,
        "debug_log_path": debug_log_path,
        "notes": f"Ingested via tool. Total PnL: {meta.get('total_pnl', 'N/A')}",
    }

    try:
        analytics.register_run(run_data)
    except Exception as e:
        logger.error(f"Failed to register run: {e}")
        # Continue? If run exists, maybe we just want to update or skip.
        # For now, assume unique run_id.
        return

    # 5. Ingest Decisions
    if os.path.exists(decisions_csv):
        logger.info("Reading decisions...")
        df_dec = pd.read_csv(decisions_csv)
        dec_records = normalize_decisions(df_dec, run_id)
        logger.info(f"Inserting {len(dec_records)} decisions...")
        analytics.bulk_insert_decisions(dec_records)
    else:
        logger.warning(f"Decisions file not found: {decisions_csv}")

    # 6. Ingest Trades
    if os.path.exists(trades_csv):
        logger.info("Reading trades...")
        df_trades = pd.read_csv(trades_csv)
        trades_records = normalize_trades(df_trades, run_id)
        logger.info(f"Inserting {len(trades_records)} trades...")
        analytics.bulk_insert_trades(trades_records)
    else:
        logger.warning(f"Trades file not found: {trades_csv}")

    # 7. Ingest Metrics (Basic)
    if "total_pnl" in meta:
        metrics = [
            {"run_id": run_id, "segment": "overall", "metric_name": "total_pnl", "metric_value": meta["total_pnl"]}
        ]
        analytics.insert_metrics(metrics)

    logger.info(f"Run {run_id} ingestion complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest backtest results into analytics DB.")
    parser.add_argument("--run-id", help="Unique ID for the run. If not provided, infers from latest report.")
    parser.add_argument("--results-dir", default="backtesting_results", help="Directory containing reports/logs.")
    parser.add_argument("--data-dir", default="data", help="Directory containing CSVs.")
    parser.add_argument("--trades-csv", help="Path to trades CSV.")
    parser.add_argument("--decisions-csv", help="Path to decisions CSV.")

    args = parser.parse_args()

    analytics = BacktestAnalytics()

    # Determine run_id and files
    if args.run_id:
        run_id = args.run_id
        # Try to find matching files
        report_files = glob.glob(os.path.join(args.results_dir, f"backtest_report_*{run_id}*.txt"))
        debug_files = glob.glob(os.path.join(args.results_dir, f"backtest_debug_*{run_id}*.log"))

        report_path = report_files[0] if report_files else None
        debug_path = debug_files[0] if debug_files else None

        # CSVs usually don't have run_id in name unless renamed.
        # Standard backtest_runner overwrites data/backtest_*.csv
        # So we assume the current files in data/ belong to this run if we are running immediately after.
        # OR we assume the user has renamed them.
        # For this tool, let's assume we are ingesting the *current* state of data/backtest_*.csv
        # and associating it with the provided run_id (or latest report timestamp).

        trades_csv = args.trades_csv or os.path.join(args.data_dir, "backtest_trades.csv")
        decisions_csv = args.decisions_csv or os.path.join(args.data_dir, "backtest_decision_log.csv")

    else:
        # Find latest report
        reports = glob.glob(os.path.join(args.results_dir, "backtest_report_*.txt"))
        if not reports:
            logger.error("No backtest reports found.")
            return

        latest_report = max(reports, key=os.path.getctime)
        report_path = latest_report

        # Extract timestamp from filename: backtest_report_YYYYMMDD_HHMMSS.txt
        # basename: backtest_report_20241202_143740.txt
        basename = os.path.basename(latest_report)
        m = re.search(r"(\d{8}_\d{6})", basename)
        if m:
            timestamp_str = m.group(1)
            run_id = timestamp_str
        else:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Find corresponding debug log
        debug_path = os.path.join(args.results_dir, f"backtest_debug_{run_id}.log")
        if not os.path.exists(debug_path):
            debug_path = None

        trades_csv = args.trades_csv or os.path.join(args.data_dir, "backtest_trades.csv")
        decisions_csv = args.decisions_csv or os.path.join(args.data_dir, "backtest_decision_log.csv")

    if not report_path:
        logger.warning("No report file found.")
        return

    ingest_run(run_id, report_path, debug_path, trades_csv, decisions_csv, analytics)


if __name__ == "__main__":
    main()
