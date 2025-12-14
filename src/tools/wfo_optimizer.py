import itertools
import logging
import multiprocessing
import os
import sys
import time

# Force single-threaded execution for ML libraries to prevent deadlocks in multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config import PAIR_CONFIGS, PairConfig
from src.core.backtest_analytics import BacktestAnalytics
from src.core.backtest_engine import BacktestEngine

# Setup Logging
# Set root logger to WARNING to suppress all module logs by default
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Configure WFO logger to INFO to show progress
logger = logging.getLogger("WFO_Optimizer")
logger.setLevel(logging.INFO)


def _worker_run_backtest(args: Tuple[PairConfig, Dict[str, Any], str, str]) -> Dict[str, Any]:
    """
    Worker function for multiprocessing.
    Runs a single backtest with specific overrides.
    """
    pair_config, overrides, start, end = args
    try:
        engine = BacktestEngine(pair_config, overrides)
        result = engine.run(start, end)

        return {
            "params": overrides,
            "sharpe": result.metrics.get("sharpe_ratio", -999.0),
            "pnl": result.metrics.get("pnl_percent", -999.0),
            "trades": result.metrics.get("total_trades", 0),
        }
    except Exception as e:
        return {"params": overrides, "sharpe": -999.0, "pnl": -999.0, "trades": 0, "error": str(e)}


class WalkForwardOptimizer:
    def __init__(
        self,
        pair_symbol: str,
        start_date: str,
        end_date: str,
        train_window_days: int = 90,
        test_window_days: int = 30,
        param_grid: Optional[Dict[str, List[Any]]] = None,
    ):
        self.pair_symbol = pair_symbol
        if pair_symbol not in PAIR_CONFIGS:
            raise ValueError(f"Unknown pair: {pair_symbol}")
        self.pair_config = PAIR_CONFIGS[pair_symbol]

        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.train_window = timedelta(days=train_window_days)
        self.test_window = timedelta(days=test_window_days)

        self.analytics = BacktestAnalytics()

        # Use provided grid or raise error
        if param_grid is None:
            raise ValueError("param_grid must be provided")
        self.grid = param_grid

    def generate_windows(self) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Generates (train_start, train_end, test_start, test_end) tuples.
        """
        windows = []
        current_test_start = self.start_date + self.train_window

        while current_test_start + self.test_window <= self.end_date:
            train_start = current_test_start - self.train_window
            train_end = current_test_start  # Exclusive in pandas slicing usually, but inclusive in logic
            test_end = current_test_start + self.test_window

            windows.append((train_start, train_end, current_test_start, test_end))
            current_test_start += self.test_window

        return windows

    def optimize_window(self, train_start: str, train_end: str) -> Dict[str, Any]:
        """
        Runs Grid Search on the training window.
        Returns the best parameter set.
        """
        keys = list(self.grid.keys())
        values = list(self.grid.values())
        combinations = list(itertools.product(*values))

        tasks = []
        for combo in combinations:
            overrides = dict(zip(keys, combo))
            # overrides["ML_ENABLED"] = False  # Removed hardcoded disable. Controlled by Config.
            tasks.append((self.pair_config, overrides, train_start, train_end))

        # Run Parallel
        # Use all CPU cores
        cpu_count = multiprocessing.cpu_count()

        with multiprocessing.Pool(processes=cpu_count) as pool:
            results = pool.map(_worker_run_backtest, tasks)

        # Filter valid results
        valid_results = [r for r in results if r["trades"] > 5]  # Min trades filter

        if not valid_results:
            logger.warning(f"No valid results for window {train_start} - {train_end}. Using default.")
            return {}

        # Sort by Sharpe
        best_result = sorted(valid_results, key=lambda x: x["sharpe"], reverse=True)[0]
        return cast(Dict[str, Any], best_result["params"])

    def run(self) -> None:
        """
        Executes the WFO loop.
        """
        windows = self.generate_windows()
        # logger.info(f"Generated {len(windows)} WFO windows.")

        wfo_trades = []
        param_history = []

        total_start = time.time()

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"Processing Window {i + 1}/{len(windows)}: Test {test_start.date()} to {test_end.date()}")

            # 1. Optimize (In-Sample)
            s_train = str(train_start)
            e_train = str(train_end)
            best_params = self.optimize_window(s_train, e_train)

            logger.info(f"Window {i + 1} [{test_start.date()} - {test_end.date()}]: Best Params = {best_params}")
            param_history.append({"window_index": i, "test_start": test_start, "params": best_params})

            # 2. Validate (Out-of-Sample)
            s_test = str(test_start)
            e_test = str(test_end)

            engine = BacktestEngine(self.pair_config, best_params)
            result = engine.run(s_test, e_test)

            # 3. Aggregate
            if result.trades:
                wfo_trades.extend(result.trades)

            # Log to DB
            self._log_wfo_run(result, "WFO_TEST", best_params)

        total_time = time.time() - total_start
        logger.info(f"WFO Complete in {total_time:.2f}s")

        # Save Results
        self._save_results(wfo_trades, param_history)

    def _log_wfo_run(self, result: Any, run_type: str, params: Dict[str, Any]) -> None:
        # Use existing analytics to log the run
        # We can add params to 'notes'
        try:
            import json

            run_data = {
                "run_id": f"WFO_{result.run_id}",
                "run_type": run_type,
                "start_date": result.start_date,
                "end_date": result.end_date,
                "strategy_version_id": None,
                "feature_engine_version_id": None,
                "data_source_id": None,
                "config_snapshot_id": self.analytics.get_or_create_config_snapshot(result.config),
                "code_version": "HEAD",
                "report_path": "",
                "debug_log_path": "",
                "notes": json.dumps(params),
            }
            self.analytics.register_run(run_data)

            # We could log trades/metrics too, but maybe overkill for every window?
            # Let's log metrics at least
            metrics_list = [
                {"run_id": f"WFO_{result.run_id}", "segment": "overall", "metric_name": k, "metric_value": v}
                for k, v in result.metrics.items()
            ]
            self.analytics.insert_metrics(metrics_list)

            # Log Trades & Decisions (Crucial for analysis)
            wfo_run_id = f"WFO_{result.run_id}"

            # Trades
            if result.trades:
                clean_trades = []
                for t in result.trades:
                    # Map to bt_trades schema
                    clean_t = {
                        "run_id": wfo_run_id,
                        "client_oid": t.get("client_oid"),
                        "timestamp": str(t.get("timestamp")) if t.get("timestamp") else None,
                        "side": t.get("side"),
                        "price": t.get("price"),
                        "amount": t.get("amount"),
                        "fee": t.get("fee"),
                        "pnl": t.get("pnl"),
                        "pnl_pct": t.get("pnl_percent"),
                        "status": t.get("status"),
                        "order_type": "MARKET",  # Default for sim
                        "extra": t.get("reason"),  # Store reason in extra
                    }
                    clean_trades.append(clean_t)
                self.analytics.bulk_insert_trades(clean_trades)

            # Decisions
            if result.decision_log:
                clean_decisions = []
                for d in result.decision_log:
                    # Flatten trade_info
                    trade_info = d.get("trade_info") or {}

                    # Map to bt_decisions schema
                    clean_d = {
                        "run_id": wfo_run_id,
                        "timestamp": str(d.get("timestamp")) if d.get("timestamp") else None,
                        "regime": d.get("regime"),
                        "action": d.get("action"),
                        "ml_score": d.get("ml_score"),
                        "close": d.get("close"),
                        "rsi": d.get("rsi"),
                        # Flattened Trade Info
                        "trade_id": trade_info.get("trade_id"),
                        "execution_price": trade_info.get("execution_price"),
                        "execution_qty": trade_info.get("execution_qty"),
                        "reason_string": trade_info.get("exit_reason"),
                    }
                    clean_decisions.append(clean_d)
                self.analytics.bulk_insert_decisions(clean_decisions)

        except Exception as e:
            logger.error(f"Failed to log WFO run: {e}")

    def _save_results(self, trades: List[Dict[str, Any]], param_history: List[Dict[str, Any]]) -> None:
        timestamp = int(time.time())

        # Save Trades
        if trades:
            df_trades = pd.DataFrame(trades)
            df_trades.to_csv(f"data/wfo_trades_{self.pair_symbol}_{timestamp}.csv", index=False)

            # Calculate WFO Metrics
            if "pnl" in df_trades.columns:
                total_pnl = df_trades["pnl"].sum()
                logger.info(f"Total WFO PnL: {total_pnl:.2f}")
            else:
                logger.warning("No PnL column found in trades (possibly only entries). Total PnL: 0.00")

        # Save Params
        df_params = pd.DataFrame(param_history)
        df_params.to_csv(f"data/wfo_params_{self.pair_symbol}_{timestamp}.csv", index=False)
        logger.info("Parameter history saved.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="SOLUSDT")
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2025-04-01")
    args = parser.parse_args()

    # Define Grids
    dca_grid: Dict[str, List[Any]] = {
        "dca_interval_minutes": [30, 60, 120, 240],
        "dca_dip_threshold_rsi": [30, 40, 50, 60],
        # "dca_notional_per_trade": [10.0, 20.0] # Optional: Test sizing
    }

    strategy_grid: Dict[str, List[Any]] = {
        "EMA_PERIOD_FAST": [15, 20, 25],
        "EMA_PERIOD_SLOW": [180, 200, 220],
        "ADX_THRESHOLD": [28, 30, 35],
        "ATR_MULTIPLIER": [2.8, 3.0, 3.5],
    }

    # Select Grid based on Config
    if args.pair in PAIR_CONFIGS and PAIR_CONFIGS[args.pair].enable_dca_mode:
        selected_grid = dca_grid
        print(f"Using DCA Grid for {args.pair}")
    else:
        selected_grid = strategy_grid
        print(f"Using Strategy Grid for {args.pair}")

    optimizer = WalkForwardOptimizer(args.pair, args.start, args.end, param_grid=selected_grid)
    optimizer.run()
