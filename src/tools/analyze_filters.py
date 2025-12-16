import sys
import pandas as pd
from src.core.backtest_analytics import BacktestAnalytics


def analyze_filters(run_id: str) -> None:
    analytics = BacktestAnalytics()
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
        # Use analytics.db.execute to get cursor, then fetchall or use pandas with connection
        # BacktestAnalytics exposes get_decisions but not arbitrary queries easily.
        # But we can access analytics.db._get_connection() if needed, or add a method.
        # Since we are in tools, accessing internal _get_connection is acceptable for now,
        # or we can use analytics.db.execute which returns a cursor.
        
        # Using pandas read_sql_query with the connection from Database class
        conn = analytics.db._get_connection()
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
        conn = analytics.db._get_connection()
        df_regime = pd.read_sql_query(query_regime, conn, params=(run_id,))
        print("\nRegime Distribution in Decisions:")
        print(df_regime)
    except Exception as e:
        print(f"Error checking regimes: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/tools/analyze_filters.py <run_id>")
    else:
        analyze_filters(sys.argv[1])
