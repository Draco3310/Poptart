import json
import os
import sqlite3
import sys

import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from typing import Optional

from src.config import Config


def analyze_wfo_results(pair_filter: Optional[str] = None) -> None:
    db_path = Config.DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # 1. Get WFO Runs
    query = """
    SELECT run_id, start_date, end_date, notes, created_at
    FROM bt_runs
    WHERE run_type = 'WFO_TEST'
    ORDER BY created_at ASC
    """

    runs = pd.read_sql_query(query, conn)

    if runs.empty:
        print("No WFO runs found.")
        return

    results = []

    print(f"Analyzing {len(runs)} WFO runs...")

    for _, run in runs.iterrows():
        run_id = run["run_id"]

        # Filter by pair if requested
        # Heuristic: run_id usually contains symbol, e.g. "WFO_XRPUSDT_..."
        if pair_filter and pair_filter not in run_id:
            continue

        # Parse Params
        try:
            params = json.loads(run["notes"])
        except Exception:
            params = {}

        # Get Metrics
        metrics_query = "SELECT metric_name, metric_value FROM bt_metrics WHERE run_id = ?"
        metrics_df = pd.read_sql_query(metrics_query, conn, params=(run_id,))
        metrics = dict(zip(metrics_df["metric_name"], metrics_df["metric_value"]))

        # Get Trades for deeper analysis
        trades_query = "SELECT * FROM bt_trades WHERE run_id = ?"
        trades_df = pd.read_sql_query(trades_query, conn, params=(run_id,))

        win_rate = 0.0
        profit_factor = 0.0

        if not trades_df.empty:
            # Ensure PnL exists (it might be missing in older runs, but we fixed it)
            if "pnl" in trades_df.columns:
                wins = trades_df[trades_df["pnl"] > 0]
                losses = trades_df[trades_df["pnl"] <= 0]

                win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0

                gross_profit = wins["pnl"].sum()
                gross_loss = abs(losses["pnl"].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        entry = {
            "run_id": run_id,
            "start_date": run["start_date"],
            "end_date": run["end_date"],
            "pnl": metrics.get("pnl", 0.0),
            "sharpe": metrics.get("sharpe_ratio", 0.0),
            "trades": metrics.get("total_trades", 0),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            **params,  # Flatten params
        }
        results.append(entry)

    if not results:
        print(f"No results found for filter: {pair_filter}")
        return

    df = pd.DataFrame(results)

    # Group by Pair (inferred from run_id)
    df["pair"] = df["run_id"].apply(lambda x: x.split("_")[1] if len(x.split("_")) > 1 else "UNKNOWN")

    for pair, group in df.groupby("pair"):
        print(f"\n=== Analysis for {pair} ===")
        print(f"Total Windows: {len(group)}")
        print(f"Total PnL: {group['pnl'].sum():.2f}")
        print(f"Average Sharpe: {group['sharpe'].mean():.2f}")
        print(f"Average Win Rate: {group['win_rate'].mean():.2%}")

        # Parameter Stability Analysis
        param_cols = [
            c
            for c in group.columns
            if c
            not in ["run_id", "start_date", "end_date", "pnl", "sharpe", "trades", "win_rate", "profit_factor", "pair"]
        ]

        print("\nParameter Distribution:")
        for col in param_cols:
            print(f"\n{col}:")
            print(group[col].value_counts(normalize=True).head(3))

        # Best Window
        best_window = group.loc[group["pnl"].idxmax()]
        print(f"\nBest Window ({best_window['start_date']} - {best_window['end_date']}):")
        print(f"PnL: {best_window['pnl']:.2f}")
        print(f"Params: {best_window[param_cols].to_dict()}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_wfo_results(sys.argv[1])
    else:
        analyze_wfo_results()
