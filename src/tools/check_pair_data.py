import argparse
import logging
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_data(file_path: str, max_gap_minutes: int = 5, jump_threshold: float = 0.20) -> None:
    """
    Checks a market data CSV for quality issues.

    Args:
        file_path: Path to the CSV file.
        max_gap_minutes: Threshold in minutes to flag a gap as significant.
        jump_threshold: Percentage change (0.20 = 20%) to flag as an extreme price jump.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {path}")
        return

    logger.info(f"Checking file: {path}")

    try:
        # Load data with mixed header detection
        # First, peek at the first row to guess format
        peek = pd.read_csv(path, nrows=1, header=None)
        first_row = peek.iloc[0]

        # Heuristic: If first row contains strings like 'timestamp' or 'open', it has a header.
        # If it contains numbers, it's likely headerless.
        has_header = False
        if isinstance(first_row[0], str) and "timestamp" in str(first_row[0]).lower():
            has_header = True
        elif isinstance(first_row[1], str) and "open" in str(first_row[1]).lower():
            has_header = True

        if has_header:
            logger.info("Detected header row.")
            df = pd.read_csv(path)
            df.columns = [c.lower().strip() for c in df.columns]
        else:
            logger.info("Detected NO header row. Assuming Kraken format (ts, o, h, l, c, v, count).")
            # Kraken format: timestamp, open, high, low, close, volume, count
            # We only need the first 6
            df = pd.read_csv(path, header=None, names=["timestamp", "open", "high", "low", "close", "volume", "count"])
            # Drop extra columns if any
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        # Check required columns
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            logger.error(f"Missing columns. Found: {df.columns}, Required: {required}")
            return

        # Parse timestamp
        # Try to infer format (int ms or ISO string)
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            # Assume ms if large int, s if small? Usually crypto data is ms.
            # Heuristic: if max > 3000000000 (year 2065 in seconds), it's probably ms
            if df["timestamp"].max() > 3000000000:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        df = df.sort_values("timestamp")

        # 1. Basic Stats
        row_count = len(df)
        min_date = df["timestamp"].min()
        max_date = df["timestamp"].max()
        duration = max_date - min_date

        logger.info(f"Rows: {row_count}")
        logger.info(f"Range: {min_date} to {max_date} ({duration})")

        # 2. Duplicates
        duplicates = df.duplicated(subset=["timestamp"]).sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate timestamps!")
        else:
            logger.info("No duplicate timestamps found.")

        # 3. Gaps
        # Calculate time diff in minutes
        df["diff_min"] = df["timestamp"].diff().dt.total_seconds() / 60.0

        # Expected diff is 1.0 for 1m candles, but let's just look for large gaps
        gaps = df[df["diff_min"] > max_gap_minutes]
        num_gaps = len(gaps)

        if num_gaps > 0:
            logger.warning(f"Found {num_gaps} gaps larger than {max_gap_minutes} minutes.")
            # Show top 5 largest gaps
            top_gaps = gaps.nlargest(5, "diff_min")
            for _, row in top_gaps.iterrows():
                logger.warning(f"  Gap: {row['diff_min']:.1f} min at {row['timestamp']}")
        else:
            logger.info(f"No gaps larger than {max_gap_minutes} minutes found.")

        # 4. NaNs
        nans = df[["open", "high", "low", "close", "volume"]].isna().sum()
        total_nans = nans.sum()
        if total_nans > 0:
            logger.warning("Found NaNs in OHLCV data:")
            for col, count in nans.items():
                if count > 0:
                    logger.warning(f"  {col}: {count}")
        else:
            logger.info("No NaNs found in OHLCV data.")

        # 5. Extreme Jumps
        # Calculate pct change of close
        df["pct_change"] = df["close"].pct_change().abs()
        jumps = df[df["pct_change"] > jump_threshold]
        num_jumps = len(jumps)

        if num_jumps > 0:
            logger.warning(f"Found {num_jumps} price jumps > {jump_threshold * 100:.0f}% in a single candle.")
            for _, row in jumps.iterrows():
                logger.warning(f"  Jump: {row['pct_change'] * 100:.1f}% at {row['timestamp']}")
        else:
            logger.info(f"No extreme price jumps > {jump_threshold * 100:.0f}% found.")

        logger.info("Check complete.")

    except Exception as e:
        logger.error(f"Error checking file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check market data CSV for quality issues.")
    parser.add_argument("--path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--max-gap", type=int, default=60, help="Max gap in minutes to tolerate before flagging")
    parser.add_argument("--jump-threshold", type=float, default=0.20, help="Price jump threshold (0.20 = 20%)")

    args = parser.parse_args()

    check_data(args.path, args.max_gap, args.jump_threshold)
