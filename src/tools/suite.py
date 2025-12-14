import argparse
import json
import logging
import os
import sys

import pandas as pd
from tabulate import tabulate  # type: ignore

# Add project root to path
sys.path.append(os.getcwd())

from src.core.backtest_analytics import BacktestAnalytics

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("AnalyticsSuite")


class AnalyticsSuite:
    def __init__(self) -> None:
        self.analytics = BacktestAnalytics()

    def _parse_extra_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expands extra_features JSON column into the DataFrame."""
        if "extra_features" not in df.columns:
            return df

        # Expand JSON
        def parse_json(x: str) -> dict:
            try:
                return json.loads(x) if x else {}
            except Exception:
                return {}

        extras = df["extra_features"].apply(parse_json)
        extras_df = pd.json_normalize(extras.tolist())

        # Join back (avoid duplicate columns)
        extras_df = extras_df.drop(columns=[c for c in extras_df.columns if c in df.columns], errors="ignore")
        return df.join(extras_df)

    def list_runs(self, limit: int = 20) -> None:
        """Lists recent backtest runs."""
        df = self.analytics.list_runs(limit=limit)
        if df.empty:
            print("No runs found.")
            return

        # Format for display
        display_cols = ["run_id", "run_type", "strategy", "start_date", "end_date", "trade_count", "total_pnl"]
        print(tabulate(df[display_cols], headers="keys", tablefmt="psql", showindex=False))

    def summary(self, run_id: str) -> None:
        """Provides a summary of a specific run."""
        run = self.analytics.get_run(run_id)
        if not run:
            print(f"Run {run_id} not found.")
            return

        print(f"\n=== Run Summary: {run_id} ===")
        print(f"Type: {run['run_type']}")
        print(f"Strategy: {run['strategy_name']} ({run['version_tag']})")
        print(f"Period: {run['start_date']} to {run['end_date']}")

        # Metrics
        metrics = self.analytics.get_metrics(run_id)
        if not metrics.empty:
            print("\n--- Key Metrics ---")
            # Pivot for nicer display
            overall = metrics[metrics["segment"] == "overall"]
            if not overall.empty:
                print(
                    tabulate(
                        overall[["metric_name", "metric_value"]], headers="keys", tablefmt="simple", showindex=False
                    )
                )

        # Trades
        trades = self.analytics.get_trades(run_id)
        if not trades.empty:
            print(f"\n--- Trade Stats ({len(trades)} trades) ---")
            wins = trades[trades["pnl"] > 0]
            losses = trades[trades["pnl"] <= 0]

            win_rate = len(wins) / len(trades) * 100
            avg_win = wins["pnl"].mean() if not wins.empty else 0
            avg_loss = losses["pnl"].mean() if not losses.empty else 0
            profit_factor = (
                abs(wins["pnl"].sum() / losses["pnl"].sum())
                if not losses.empty and losses["pnl"].sum() != 0
                else float("inf")
            )

            print(f"Win Rate:      {win_rate:.2f}%")
            print(f"Profit Factor: {profit_factor:.2f}")
            print(f"Avg Win:       {avg_win:.2f}")
            print(f"Avg Loss:      {avg_loss:.2f}")
            print(f"Total PnL:     {trades['pnl'].sum():.2f}")

    def diagnose(self, run_id: str) -> None:
        """Diagnoses failures and skipped trades."""
        print(f"\n=== Diagnosis: {run_id} ===")

        # 1. Filter Reasons
        decisions = self.analytics.get_decisions(run_id)
        if decisions.empty:
            print("No decisions found.")
            return

        decisions = self._parse_extra_features(decisions)

        print(f"Total Decisions: {len(decisions)}")

        # Group by Action
        action_counts = decisions["action"].value_counts()
        print("\n--- Action Distribution ---")
        print(action_counts)

        # Analyze Skips
        skips = decisions[decisions["action"].isin(["CANDIDATE_IGNORED", "SIGNAL_FILTERED", "SKIP"])]
        if not skips.empty:
            print("\n--- Top Skip Reasons ---")
            reasons = skips["reason_string"].value_counts().head(10)
            print(reasons)

            # Analyze BTC Filter Impact
            if "btc_bullish" in skips.columns:
                print("\n--- BTC Filter Impact (Skips) ---")
                print(skips["btc_bullish"].value_counts(dropna=False))

            # Analyze Volume Gate Impact
            if "vol_ratio" in skips.columns:
                print("\n--- Volume Ratio Stats (Skips) ---")
                print(skips["vol_ratio"].describe())

        # Analyze Regime Distribution
        print("\n--- Regime Distribution ---")
        regimes = decisions["regime"].value_counts()
        print(regimes)

        # Analyze ML Scores (if available)
        if "ml_score" in decisions.columns and decisions["ml_score"].notna().any():
            print("\n--- ML Score Stats ---")
            print(decisions["ml_score"].describe())

    def performance(self, run_id: str) -> None:
        """Analyzes performance drivers."""
        print(f"\n=== Performance Analysis: {run_id} ===")

        trades = self.analytics.get_trades(run_id)
        decisions = self.analytics.get_decisions(run_id)

        if trades.empty or decisions.empty:
            print("Insufficient data.")
            return

        decisions = self._parse_extra_features(decisions)

        # Filter for PnL generating trades (Exits)
        exits = trades[trades["pnl"] != 0].copy()

        # Reconstruct Round Trips
        round_trips = []

        for _, row in exits.iterrows():
            related_oid = None
            if row.get("extra"):
                try:
                    extra_json = json.loads(row["extra"]) if isinstance(row["extra"], str) else row["extra"]
                    related_oid = extra_json.get("related_oid")
                except Exception:
                    pass

            if not related_oid:
                continue

            # Find entry decision
            entry_decision = decisions[decisions["trade_id"] == related_oid]
            if not entry_decision.empty:
                d = entry_decision.iloc[0]
                rt = {
                    "pnl": row["pnl"],
                    "pnl_pct": row["pnl_pct"],
                    "exit_reason": row.get("extra"),
                    "rsi": d["rsi"],
                    "adx": d["adx"],
                    "regime": d["regime"],
                    "ml_score": d["ml_score"],
                    "btc_bullish": d.get("btc_bullish"),
                    "vol_ratio": d.get("vol_ratio"),
                    "ema_extension": d.get("ema_extension"),
                }
                round_trips.append(rt)

        if not round_trips:
            print("Could not reconstruct round trips.")
            return

        df_rt = pd.DataFrame(round_trips)

        # PnL by Regime
        print("\n--- PnL by Regime ---")
        print(df_rt.groupby("regime")["pnl"].agg(["count", "sum", "mean"]))

        # PnL by RSI Bucket
        df_rt["rsi_bucket"] = pd.cut(df_rt["rsi"], bins=[0, 30, 40, 50, 60, 70, 80, 100])
        print("\n--- PnL by RSI ---")
        print(df_rt.groupby("rsi_bucket", observed=True)["pnl"].agg(["count", "sum", "mean"]))

        # PnL by ADX Bucket
        if "adx" in df_rt.columns:
            df_rt["adx_bucket"] = pd.cut(df_rt["adx"], bins=[0, 15, 25, 35, 50, 100])
            print("\n--- PnL by ADX ---")
            print(df_rt.groupby("adx_bucket", observed=True)["pnl"].agg(["count", "sum", "mean"]))

        # PnL by Volume Ratio
        if "vol_ratio" in df_rt.columns:
            df_rt["vol_bucket"] = pd.cut(df_rt["vol_ratio"], bins=[0, 0.5, 1.0, 1.2, 2.0, 5.0, 100])
            print("\n--- PnL by Volume Ratio ---")
            print(df_rt.groupby("vol_bucket", observed=True)["pnl"].agg(["count", "sum", "mean"]))

        # PnL by BTC Trend
        if "btc_bullish" in df_rt.columns:
            print("\n--- PnL by BTC Trend ---")
            print(df_rt.groupby("btc_bullish")["pnl"].agg(["count", "sum", "mean"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Poptart Analytics Suite")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List
    parser_list = subparsers.add_parser("list", help="List recent runs")
    parser_list.add_argument("--limit", type=int, default=20, help="Number of runs to show")

    # Summary
    parser_summary = subparsers.add_parser("summary", help="Show run summary")
    parser_summary.add_argument("run_id", type=str, help="Run ID")

    # Diagnose
    parser_diagnose = subparsers.add_parser("diagnose", help="Diagnose failures")
    parser_diagnose.add_argument("run_id", type=str, help="Run ID")

    # Performance
    parser_perf = subparsers.add_parser("performance", help="Analyze performance")
    parser_perf.add_argument("run_id", type=str, help="Run ID")

    args = parser.parse_args()

    suite = AnalyticsSuite()

    if args.command == "list":
        suite.list_runs(args.limit)
    elif args.command == "summary":
        suite.summary(args.run_id)
    elif args.command == "diagnose":
        suite.diagnose(args.run_id)
    elif args.command == "performance":
        suite.performance(args.run_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
