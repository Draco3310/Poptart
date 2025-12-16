import sys
import pandas as pd
from src.core.backtest_analytics import BacktestAnalytics


def analyze_trades(run_id: str) -> None:
    analytics = BacktestAnalytics()
    conn = analytics.db._get_connection()

    # 1. Get All Trades
    query = f"SELECT * FROM bt_trades WHERE run_id = '{run_id}' ORDER BY timestamp"
    trades = pd.read_sql(query, conn)

    if trades.empty:
        print("No trades found.")
        return

    # 2. Reconstruct Round Trips
    entries = {}  # client_oid -> trade_row
    round_trips = []

    for _, row in trades.iterrows():
        if pd.isna(row["pnl"]) or row["pnl"] == 0:
            if row["client_oid"]:
                entries[row["client_oid"]] = row
        else:
            # Exit
            related_oid = None
            if row["extra"]:
                try:
                    extra = eval(row["extra"])
                    related_oid = extra.get("related_oid")
                except Exception:
                    pass

            if related_oid and related_oid in entries:
                entry = entries[related_oid]
                round_trips.append(
                    {
                        "entry_time": entry["timestamp"],
                        "exit_time": row["timestamp"],
                        "pnl": row["pnl"],
                        "pnl_pct": row["pnl_pct"],
                        "trade_id": related_oid,
                        "exit_reason": row.get("reason", "Unknown"),
                    }
                )

    if not round_trips:
        print("No round-trip trades found.")
        return

    df = pd.DataFrame(round_trips)

    # Split into Wins and Losses
    wins = df[df["pnl"] > 0].sort_values("pnl", ascending=False)
    losses = df[df["pnl"] <= 0].sort_values("pnl", ascending=True)

    print(f"Total Trades: {len(df)}")
    print(f"Wins: {len(wins)} | Losses: {len(losses)}")
    if len(df) > 0:
        print(f"Win Rate: {len(wins) / len(df):.2%}")
        print(f"Avg Win: {wins['pnl'].mean():.2f} | Avg Loss: {losses['pnl'].mean():.2f}")

    # 3. Analyze Top 5 Losses
    print("\n--- Top 5 Losses ---")
    analyze_subset(losses.head(5), run_id, conn)

    # 4. Analyze Top 5 Wins
    print("\n--- Top 5 Wins ---")
    analyze_subset(wins.head(5), run_id, conn)


def analyze_subset(subset_df: pd.DataFrame, run_id: str, conn) -> None:
    if subset_df.empty:
        print("None.")
        return

    for i, trade in subset_df.iterrows():
        trade_id = trade["trade_id"]
        pnl_pct = trade["pnl_pct"] if trade["pnl_pct"] is not None else 0.0
        print(f"\nTrade {trade_id} | PnL: {trade['pnl']:.2f} ({pnl_pct:.2f}%)")
        print(f"Entry: {trade['entry_time']} | Exit: {trade['exit_time']}")

        # Fetch Decision
        d_query = f"""
        SELECT * FROM bt_decisions
        WHERE run_id = '{run_id}'
        AND trade_id = '{trade_id}'
        """
        decisions = pd.read_sql(d_query, conn)

        if not decisions.empty:
            row = decisions.iloc[0]
            print(f"Regime: {row['regime']}")
            print(f"Action: {row['action']}")
            print(f"Reason: {row['reason_string']}")
            print(f"Close: {row['close']:.4f}")
            print(f"EMA200: {row['ema200']:.4f}")
            print(f"EMA200_1H: {row['ema200_1h']:.4f}")
            print(f"ADX: {row['adx']:.2f}")
            rsi = row["rsi"] if row["rsi"] is not None else 0.0
            print(f"RSI: {rsi:.2f}")
            atr = row["atr"] if row["atr"] is not None else 0.0
            print(f"ATR: {atr:.4f}")
            print(f"ML Score: {row['ml_score']}")
        else:
            print("No decision log found for this trade ID.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/tools/analyze_losses.py <run_id>")
    else:
        analyze_trades(sys.argv[1])
