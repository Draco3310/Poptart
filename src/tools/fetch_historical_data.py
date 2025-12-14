import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd

# Add project root to path to import Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config import Config

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DataFetcher")


def fetch_data(days: int = 90) -> None:
    """
    Fetches historical OHLCV data from Kraken.
    """
    exchange = ccxt.kraken({"enableRateLimit": True})

    symbol = Config.SYMBOL
    timeframe = Config.TIMEFRAME_PRIMARY

    # Calculate start time
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)
    since = int(start_time.timestamp() * 1000)

    logger.info(f"Fetching {timeframe} data for {symbol} since {start_time.isoformat()}...")

    all_ohlcv = []

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)

            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)

            # Update 'since' to the timestamp of the last candle + 1ms to avoid duplicates
            # ohlcv is [timestamp, open, high, low, close, volume]
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1

            logger.info(f"Fetched {len(ohlcv)} candles. Last timestamp: {pd.to_datetime(last_timestamp, unit='ms')}")

            # Break if we've reached the current time (approx)
            if last_timestamp >= int(end_time.timestamp() * 1000) - (60 * 60 * 1000):  # Within last hour
                break

            # Rate limit sleep is handled by enableRateLimit=True in ccxt, but a small sleep is safe
            time.sleep(exchange.rateLimit / 1000)

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            break

    if not all_ohlcv:
        logger.warning("No data fetched.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Remove duplicates just in case
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp")

    # Save to CSV
    output_path = os.path.join("data", "historical_data.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} rows to {output_path}")

    # Verify
    logger.info(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")


if __name__ == "__main__":
    fetch_data()
