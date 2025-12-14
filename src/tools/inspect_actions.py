import logging

from src.core.backtest_analytics import BacktestAnalytics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inspect_actions(run_id: str) -> None:
    analytics = BacktestAnalytics()

    # We can't easily get distinct values via the API, so let's get a slice without filters (limit 1000)
    # and see what we find, or just get all decisions if memory allows (might be large).
    # Better: use the underlying DB connection if possible, but BacktestAnalytics abstracts it.
    # Let's try to get a slice with a known common action like 'SIGNAL_FILTERED' or just get the first 100 rows.

    # Actually, let's just get the first 100 rows of ANY action.
    try:
        # Passing empty filters
        df = analytics.get_decisions(run_id, limit=100)
        if not df.empty:
            logger.info(f"First 100 rows actions: {df['action'].unique()}")
            logger.info(f"First 100 rows regimes: {df['regime'].unique()}")
            logger.info(f"Sample row:\n{df.iloc[0]}")
        else:
            logger.warning("No decisions found.")

        # Also try to find where trades happened.
        trades = analytics.get_trades(run_id)
        if not trades.empty:
            logger.info(f"Found {len(trades)} trades.")
            logger.info(f"Trade sample:\n{trades.iloc[0]}")

            # Get timestamps of trades
            trade_timestamps = trades["timestamp"].tolist()
            logger.info(f"Trade timestamps: {trade_timestamps[:5]}")

            # Now try to find decisions around these timestamps
            # We can't query by timestamp range easily via get_decisions, but we can try to match exact timestamp
            # if we knew it. But decisions might be slightly before trade execution.

    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    inspect_actions("20251202_173113")
