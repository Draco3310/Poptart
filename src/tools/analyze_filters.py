import sqlite3
import sys

import pandas as pd


def analyze_filters(run_id: str) -> None:
    db_path = "data/sentinel.db"
    conn = sqlite3.connect(db_path)

    print(f"Analyzing filters for Run ID: {run_id}")

    # 1. Count reasons for SKIP/FILTERED
    query = """
    SELECT action, reason_string, COUNT(*) as count
    FROM bt_decisions
    WHERE run_id=? AND action IN ('SKIP', 'CANDIDATE_IGNORED', 'SIGNAL_FILTERED')
    GROUP BY action, reason_string
    ORDER BY count DESC
    LIMIT 20
    """

    try:
        df = pd.read_sql_query(query, conn, params=(run_id,))
        print("\nTop 20 Skip Reasons:")
        for index, row in df.iterrows():
            print(f"{row['count']} | {row['action']} | {row['reason_string']}")
    except Exception as e:
        print(f"Error: {e}")

    # 2. Check Regime Distribution in Decisions
    query_regime = """
    SELECT regime, COUNT(*) as count
    FROM bt_decisions
    WHERE run_id=?
    GROUP BY regime
    """
    try:
        df_regime = pd.read_sql_query(query_regime, conn, params=(run_id,))
        print("\nRegime Distribution in Decisions:")
        print(df_regime)
    except Exception as e:
        print(f"Error checking regimes: {e}")

    conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/tools/analyze_filters.py <run_id>")
    else:
        analyze_filters(sys.argv[1])
