import logging
import os
import sys
from datetime import datetime
from typing import List, Optional

import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.config import PAIR_CONFIGS, PairConfig, get_data_path
from src.core.backtest_analytics import BacktestAnalytics
from src.core.backtest_engine import BacktestEngine
from src.core.backtest_result import BacktestResult
from src.simulated_exchange import SimulatedKrakenExchange

# Setup Logging
# Set root logger to WARNING to suppress all module logs by default
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Configure BacktestRunner logger to INFO to show progress
logger = logging.getLogger("BacktestRunner")
logger.setLevel(logging.INFO)


def run_backtest(
    pair_config: PairConfig, start_date: Optional[str] = None, end_date: Optional[str] = None, log_to_db: bool = True
) -> BacktestResult:
    """
    Orchestrates a single backtest run using the BacktestEngine.
    Handles reporting and database ingestion.
    """
    logger.info(f"Starting Backtest for {pair_config.symbol} ({start_date} to {end_date})...")

    # 1. Load BTC Data (Context) if not running BTC
    btc_data = None
    if pair_config.symbol != "BTCUSDT":
        try:
            btc_path = get_data_path("BTCUSDT", "1m")
            if os.path.exists(btc_path):
                logger.info(f"Loading BTC Context from {btc_path}...")
                # Use SimulatedKrakenExchange to load and clean data
                btc_exchange = SimulatedKrakenExchange(str(btc_path))
                btc_data = btc_exchange.data
            else:
                logger.warning("BTC Data not found. Running without BTC Context.")
        except Exception as e:
            logger.error(f"Failed to load BTC Context: {e}")

    # 2. Run Engine
    engine = BacktestEngine(pair_config, btc_data=btc_data)
    result = engine.run(start_date, end_date)

    if result.run_id == "failed":
        logger.error("Backtest failed.")
        return result

    # 2. Generate Reports
    _generate_reports(result)

    # 3. Ingest to Database
    if log_to_db:
        _ingest_to_db(result)

    return result


def _generate_reports(result: BacktestResult) -> None:
    """Generates Console, TXT, and CSV reports."""
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "backtesting_results"
    os.makedirs(results_dir, exist_ok=True)

    report_file = os.path.join(results_dir, f"backtest_report_{result.pair}_{timestamp_str}.txt")

    # Console Output
    lines = []
    lines.append("=" * 50)
    lines.append(f"BACKTEST RESULTS: {result.pair}")
    lines.append("=" * 50)
    lines.append(f"Run ID: {result.run_id}")
    lines.append(f"PnL: {result.metrics.get('pnl', 0.0):.2f} ({result.metrics.get('pnl_percent', 0.0):.2f}%)")
    lines.append(f"Sharpe Ratio: {result.metrics.get('sharpe_ratio', 0.0):.4f}")
    lines.append(f"Total Trades: {result.metrics.get('total_trades', 0)}")
    lines.append("=" * 50)

    for line in lines:
        print(line)

    # Write Report File
    with open(report_file, "w") as f:
        f.write("\n".join(lines))

    # Write CSVs
    if result.trades:
        pd.DataFrame(result.trades).to_csv(f"data/backtest_trades_{result.pair}.csv", index=False)

    if result.decision_log:
        pd.DataFrame(result.decision_log).to_csv(f"data/backtest_decision_log_{result.pair}.csv", index=False)

    # Append to Summary CSV
    summary_file = "data/backtest_results.csv"
    summary_row = {
        "timestamp": datetime.now(),
        "symbol": result.pair,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "pnl": result.metrics.get("pnl"),
        "pnl_percent": result.metrics.get("pnl_percent"),
        "sharpe_ratio": result.metrics.get("sharpe_ratio"),
        "total_trades": result.metrics.get("total_trades"),
        "run_id": result.run_id,
    }
    pd.DataFrame([summary_row]).to_csv(summary_file, mode="a", header=not os.path.exists(summary_file), index=False)


def _ingest_to_db(result: BacktestResult) -> None:
    """Ingests the result into sentinel.db."""
    try:
        analytics = BacktestAnalytics()

        # 1. Register Run
        run_data = {
            "run_id": result.run_id,
            "run_type": "BACKTEST",
            "start_date": result.start_date,
            "end_date": result.end_date,
            "strategy_version_id": None,  # TODO: Versioning
            "feature_engine_version_id": None,
            "data_source_id": None,
            "config_snapshot_id": analytics.get_or_create_config_snapshot(result.config),
            "code_version": "HEAD",
            "report_path": "",
            "debug_log_path": "",
            "notes": "Automated Run",
        }
        analytics.register_run(run_data)

        # 2. Ingest Trades & Decisions
        # Add run_id to all records
        for t in result.trades:
            t["run_id"] = result.run_id
        for d in result.decision_log:
            d["run_id"] = result.run_id

        analytics.bulk_insert_trades(result.trades)
        analytics.bulk_insert_decisions(result.decision_log)

        # 3. Ingest Metrics
        metrics_list = [
            {"run_id": result.run_id, "segment": "overall", "metric_name": k, "metric_value": v}
            for k, v in result.metrics.items()
        ]
        analytics.insert_metrics(metrics_list)

        logger.info(f"Ingested run {result.run_id} into DB.")

    except Exception as e:
        logger.error(f"DB Ingestion failed: {e}")


def get_data_range(pair_config: PairConfig) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Efficiently determines the start and end timestamps of the data file.
    """
    data_path = str(get_data_path(pair_config.symbol, "1m"))
    if not os.path.exists(data_path):
        return None, None

    try:
        temp_exchange = SimulatedKrakenExchange(data_path)
        return temp_exchange.data["timestamp"].min(), temp_exchange.data["timestamp"].max()
    except Exception as e:
        logger.error(f"Failed to get data range for {pair_config.symbol}: {e}")
        return None, None


def run_portfolio_backtest(pairs: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None) -> None:
    """
    Runs backtests for a list of pairs sequentially.
    """
    # 1. Compute Intersection if needed
    if start_date is None or end_date is None:
        logger.info("Computing Portfolio Intersection...")
        global_start = pd.Timestamp.min
        global_end = pd.Timestamp.max

        valid_pairs = []

        for symbol in pairs:
            if symbol not in PAIR_CONFIGS:
                logger.warning(f"Skipping unknown pair: {symbol}")
                continue

            p_start, p_end = get_data_range(PAIR_CONFIGS[symbol])
            if p_start is None or p_end is None:
                logger.warning(f"Could not determine data range for {symbol}")
                continue

            logger.info(f"  {symbol}: {p_start} to {p_end}")

            if global_start == pd.Timestamp.min or p_start > global_start:
                global_start = p_start
            if global_end == pd.Timestamp.max or p_end < global_end:
                global_end = p_end

            valid_pairs.append(symbol)

        if not valid_pairs:
            logger.error("No valid pairs found for portfolio backtest.")
            return

        if global_start >= global_end:
            logger.error(f"No overlapping time window found! Start ({global_start}) >= End ({global_end})")
            return

        logger.info(f"Portfolio Intersection Window: {global_start} to {global_end}")

        if start_date is None:
            start_date = str(global_start)
        if end_date is None:
            end_date = str(global_end)

    # 2. Run Backtests
    for symbol in pairs:
        if symbol in PAIR_CONFIGS:
            logger.info(f"=== Running Backtest for {symbol} ===")
            try:
                run_backtest(PAIR_CONFIGS[symbol], start_date, end_date)
            except Exception as e:
                logger.error(f"Failed to run backtest for {symbol}: {e}")
        else:
            logger.error(f"Unknown pair: {symbol}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, help="Single pair to run (e.g. SOLUSDT)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--portfolio", action="store_true", help="Run full portfolio backtest")
    args = parser.parse_args()

    if args.portfolio:
        logger.info(">>> STARTING PORTFOLIO BACKTEST <<<")
        pairs_to_run = ["BTCUSDT", "XRPUSDT", "SOLUSDT"]
        run_portfolio_backtest(pairs_to_run, start_date=args.start or "2022-10-01", end_date=args.end or "2023-12-31")
    elif args.pair:
        if args.pair not in PAIR_CONFIGS:
            logger.error(f"Unknown pair: {args.pair}")
            sys.exit(1)
        run_backtest(PAIR_CONFIGS[args.pair], start_date=args.start, end_date=args.end)
    else:
        # Default behavior if no args provided (or just run portfolio as fallback)
        logger.info("No arguments provided. Defaulting to Portfolio Backtest.")
        pairs_to_run = ["BTCUSDT", "XRPUSDT", "SOLUSDT"]
        run_portfolio_backtest(pairs_to_run, start_date="2022-10-01", end_date="2023-12-31")
