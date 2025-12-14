import logging

import numpy as np
import pandas as pd

from src.core.backtest_analytics import BacktestAnalytics

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def analyze_performance(run_id: str) -> None:
    logger.info(f"Analyzing performance for run_id: {run_id}")

    analytics = BacktestAnalytics()

    try:
        decisions = analytics.get_decisions(run_id)
        trades = analytics.get_trades(run_id)
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return

    if decisions.empty or trades.empty:
        logger.warning("No decisions or trades found.")
        return

    # Merge decisions with trades
    # decisions has 'trade_id', trades has 'client_oid' (which matches trade_id)
    decisions.merge(trades, left_on="trade_id", right_on="client_oid", how="inner", suffixes=("", "_trade"))

    # Filter for Mean Reversion Entries (LONG)
    # We look for SELL trades to get PnL, but we want the ENTRY context.
    # The 'merged' dataframe joins the ENTRY decision with the ENTRY trade.
    # But PnL is on the SELL trade.

    # Wait, 'trades' table has PnL on the SELL row.
    # The ENTRY row has PnL = 0.
    # We need to link the ENTRY decision to the SELL trade PnL.

    # In 'backtest_trades.csv', we don't have a direct link between Entry Trade ID and Exit Trade ID.
    # But we can assume FIFO or match by sequence.
    # However, 'ingest_backtest_results.py' computes PnL on the SELL row.

    # Let's look at 'trades' dataframe structure from analytics.
    # It has 'pnl', 'pnl_pct'.

    # Strategy:
    # 1. Identify Round Trips.
    #    Since we don't have a trade_id link between buy and sell in the CSV,
    #    we might need to rely on the fact that we only hold one position at a time.
    #    So we can sort by time and pair them up.

    trades_sorted = trades.sort_values("timestamp")

    round_trips = []
    current_entry = None

    for _, row in trades_sorted.iterrows():
        if row["side"] == "BUY":
            current_entry = row
        elif row["side"] == "SELL" and current_entry is not None:
            # Found a pair
            # Find the decision for the ENTRY
            entry_decision = decisions[decisions["trade_id"] == current_entry["client_oid"]]

            if not entry_decision.empty:
                d = entry_decision.iloc[0]
                rt = {
                    "entry_time": current_entry["timestamp"],
                    "exit_time": row["timestamp"],
                    "pnl": row["pnl"],
                    "pnl_pct": row["pnl_pct"],
                    "rsi": d["rsi"],
                    "adx": d["adx"],
                    "atr": d["atr"],
                    "bb_mid": d["bb_mid"],
                    "close": d["close"],  # Entry price (approx) or decision price
                    "obi": d["obi"],
                    "spread": d["spread"],
                    "market_depth_ratio": d["market_depth_ratio"],
                    "reason": d["reason_string"],
                }

                # Derived metrics
                if rt["close"] and rt["bb_mid"]:
                    rt["dist_to_mean"] = (rt["bb_mid"] - rt["close"]) / rt["close"]
                else:
                    rt["dist_to_mean"] = np.nan

                if rt["close"] and rt["atr"]:
                    rt["volatility"] = rt["atr"] / rt["close"]
                else:
                    rt["volatility"] = np.nan

                round_trips.append(rt)

            current_entry = None  # Reset

    df_rt = pd.DataFrame(round_trips)

    if df_rt.empty:
        logger.warning("No round trips identified.")
        return

    logger.info(f"Identified {len(df_rt)} round trips.")
    logger.info(f"Total PnL: {df_rt['pnl'].sum():.2f}")

    # Analysis 1: RSI Buckets
    logger.info("\n--- Performance by RSI Bucket ---")
    df_rt["rsi_bucket"] = pd.cut(df_rt["rsi"], bins=[0, 20, 25, 30, 35, 40, 50, 100])
    rsi_stats = df_rt.groupby("rsi_bucket", observed=True)["pnl"].agg(["count", "sum", "mean"])
    print(rsi_stats)

    # Analysis 2: Distance to Mean Buckets
    logger.info("\n--- Performance by Distance to Mean ---")
    # dist_to_mean is usually positive for Longs (price < bb_mid)
    df_rt["dist_bucket"] = pd.cut(df_rt["dist_to_mean"], bins=[-0.01, 0.0, 0.005, 0.01, 0.015, 0.02, 0.05, 1.0])
    dist_stats = df_rt.groupby("dist_bucket", observed=True)["pnl"].agg(["count", "sum", "mean"])
    print(dist_stats)

    # Analysis 3: Volatility (ATR/Close)
    logger.info("\n--- Performance by Volatility (ATR/Close) ---")
    df_rt["vol_bucket"] = pd.cut(df_rt["volatility"], bins=[0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.05])
    vol_stats = df_rt.groupby("vol_bucket", observed=True)["pnl"].agg(["count", "sum", "mean"])
    print(vol_stats)

    # Analysis 4: OBI (Order Book Imbalance)
    if "obi" in df_rt.columns and df_rt["obi"].notna().any():
        logger.info("\n--- Performance by OBI ---")
        df_rt["obi_bucket"] = pd.cut(df_rt["obi"], bins=[-1, -0.5, -0.2, 0, 0.2, 0.5, 1])
        obi_stats = df_rt.groupby("obi_bucket", observed=True)["pnl"].agg(["count", "sum", "mean"])
        print(obi_stats)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="Run ID to analyze")
    args = parser.parse_args()

    analyze_performance(args.run_id)
