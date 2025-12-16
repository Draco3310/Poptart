import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads OHLCV data from a CSV file.
    Handles header detection, column normalization, and timestamp conversion.
    """
    logger.info(f"Loading data from {filepath}...")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # Check for header
    try:
        peek = pd.read_csv(filepath, nrows=1, header=None)
    except pd.errors.EmptyDataError:
        logger.warning(f"Empty file: {filepath}")
        return pd.DataFrame()

    first_val = str(peek.iloc[0, 0])

    if "timestamp" in first_val.lower() or "open_time" in first_val.lower():
        # Has header
        df = pd.read_csv(filepath)
        # Normalize columns
        df.columns = [c.lower().strip() for c in df.columns]

        # Map 'open_time' to 'timestamp' if needed
        if "open_time" in df.columns and "timestamp" not in df.columns:
            df.rename(columns={"open_time": "timestamp"}, inplace=True)

    else:
        # No header, assume Kraken format
        # timestamp, open, high, low, close, volume, trades
        df = pd.read_csv(filepath, header=None, names=["timestamp", "open", "high", "low", "close", "volume", "trades"])

    # Downcast numeric columns to float32
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    # Convert timestamp
    if "timestamp" in df.columns:
        # Check if it's already datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            # Check if it's ms or s
            # Heuristic: 30000000000 is roughly year 2920 in seconds, so anything larger is likely ms
            if df["timestamp"].max() > 30000000000:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

    logger.info(f"Loaded {len(df)} rows.")
    return df
