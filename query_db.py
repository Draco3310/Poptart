import sqlite3

import pandas as pd

db_path = "data/sentinel.db"
run_id = "BTCUSDT_20251205_091706"

conn = sqlite3.connect(db_path)

print(f"Analyzing Run: {run_id}")

# 1. Count trades by side
query_counts = """
SELECT side, COUNT(*) as count
FROM bt_trades
WHERE run_id = ?
GROUP BY side
"""
df_counts = pd.read_sql_query(query_counts, conn, params=(run_id,))
print("\nTrade Counts by Side:")
print(df_counts)

# 2. Check first few trades to see frequency
query_trades = """
SELECT timestamp, side, price, amount, fee
FROM bt_trades
WHERE run_id = ?
ORDER BY timestamp ASC
LIMIT 20
"""
df_trades = pd.read_sql_query(query_trades, conn, params=(run_id,))
print("\nFirst 20 Trades:")
print(df_trades)

# 3. Check decisions for these trades
# We want to see why it's buying/selling.
query_decisions = """
SELECT timestamp, action, reason_string, ml_score, rsi, bb_lower, close
FROM bt_decisions
WHERE run_id = ?
ORDER BY timestamp ASC
LIMIT 20
"""
df_decisions = pd.read_sql_query(query_decisions, conn, params=(run_id,))
print("\nFirst 20 Decisions:")
print(df_decisions)

conn.close()
