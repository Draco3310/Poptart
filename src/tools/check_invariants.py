import logging

import pandas as pd

from src.config import Config
from src.core.backtest_analytics import BacktestAnalytics

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_invariants(run_id: str) -> None:
    logger.info(f"Checking invariants for run_id: {run_id}")

    analytics = BacktestAnalytics()

    # Fetch all decisions for the run
    # We want to look at entries in the CHOP regime
    try:
        decisions = analytics.get_decisions(run_id)
    except Exception as e:
        logger.error(f"Failed to fetch decisions: {e}")
        return

    if decisions.empty:
        logger.warning("No decisions found for this run.")
        return

    # Filter for Mean Reversion entries
    # Note: StrategySelector is currently forcing Mean Reversion, so we check all entries.
    # We don't filter by regime='CHOP' because RegimeClassifier might say TREND (ADX > 25)
    # while MeanReversionStrategy still trades (ADX < 50).
    mr_entries = decisions[decisions["action"].str.contains("ENTRY")].copy()

    logger.info(f"Found {len(mr_entries)} entries (checking against Mean Reversion logic).")

    if mr_entries.empty:
        return

    violations = []

    for idx, row in mr_entries.iterrows():
        action = row["action"]
        timestamp = row["timestamp"]

        # Common Invariants
        if row["is_ranging"] != 1:
            violations.append(f"{timestamp} [{action}]: is_ranging is {row['is_ranging']} (expected 1)")

        if row["touched_band"] != 1:
            violations.append(f"{timestamp} [{action}]: touched_band is {row['touched_band']} (expected 1)")

        # LONG Invariants
        if action == "ENTRY_LONG":
            if row["is_uptrend_1h"] != 1:
                violations.append(f"{timestamp} [{action}]: is_uptrend_1h is {row['is_uptrend_1h']} (expected 1)")

            if row["rsi"] >= Config.RSI_OVERSOLD:
                violations.append(f"{timestamp} [{action}]: RSI {row['rsi']:.2f} >= {Config.RSI_OVERSOLD}")

            # Check close vs bb_mid (should be below)
            # Note: bb_mid might be missing if not logged, but it is in decision_context
            if pd.notna(row.get("bb_mid")) and row["close"] >= row["bb_mid"]:
                violations.append(f"{timestamp} [{action}]: Close {row['close']:.4f} >= BB_Mid {row['bb_mid']:.4f}")

        # SHORT Invariants
        elif action == "ENTRY_SHORT":
            if row["is_downtrend_1h"] != 1:
                violations.append(f"{timestamp} [{action}]: is_downtrend_1h is {row['is_downtrend_1h']} (expected 1)")

            if row["rsi"] <= Config.RSI_OVERBOUGHT:
                violations.append(f"{timestamp} [{action}]: RSI {row['rsi']:.2f} <= {Config.RSI_OVERBOUGHT}")

            if pd.notna(row.get("bb_mid")) and row["close"] <= row["bb_mid"]:
                violations.append(f"{timestamp} [{action}]: Close {row['close']:.4f} <= BB_Mid {row['bb_mid']:.4f}")

    if violations:
        logger.error(f"Found {len(violations)} invariant violations:")
        for v in violations:
            logger.error(v)
    else:
        logger.info("✅ No invariant violations found for Mean Reversion entries.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="20251202_173113", help="Run ID to check")
    args = parser.parse_args()

    check_invariants(args.run_id)
