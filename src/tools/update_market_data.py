import logging
import os
import sys
import time
from datetime import datetime

import ccxt
import pandas as pd
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config import SYMBOL_MAP

# Load environment variables
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MarketDataUpdater")

# Configuration
PAIRS = ["XRP/USDT", "SOL/USDT", "BTC/USDT"]


def update_pair(pair: str) -> None:
    symbol_clean = pair.replace("/", "")
    coin_dir = SYMBOL_MAP.get(symbol_clean, symbol_clean)
    csv_path = os.path.join("data", str(coin_dir), f"{symbol_clean}_1m.csv")

    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}. Please run download_binance_data.py first.")
        return

    logger.info(f"Updating {pair} from {csv_path}...")

    # Load existing data
    df = pd.read_csv(csv_path)
    if df.empty:
        logger.error("CSV is empty.")
        return

    # Get last timestamp
    last_ts = int(df.iloc[-1]["open_time"])
    last_date = datetime.fromtimestamp(last_ts / 1000)
    logger.info(f"Last data point: {last_date}")

    # Initialize Exchange
    exchange = ccxt.kraken()

    # Fetch Data
    new_candles = []
    since = last_ts + 60000  # Start from next minute

    while True:
        try:
            # Log progress
            current_date = datetime.fromtimestamp(since / 1000)
            if current_date > datetime.now():
                break

            logger.info(f"Fetching from {current_date}...")

            candles = exchange.fetch_ohlcv(pair, timeframe="1m", since=since, limit=720)

            if not candles:
                break

            new_candles.extend(candles)

            # Update since
            last_candle_ts = candles[-1][0]
            if last_candle_ts <= since:
                break  # Prevent infinite loop if no new data

            since = last_candle_ts + 60000

            # Rate limit
            time.sleep(1.0)

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            break

    if not new_candles:
        logger.info("No new data found.")
        return

    logger.info(f"Downloaded {len(new_candles)} new candles.")

    # Convert to DataFrame
    new_df = pd.DataFrame(new_candles, columns=["open_time", "open", "high", "low", "close", "volume"])

    # Add date column (as per download_binance_data.py format)
    new_df["date"] = pd.to_datetime(new_df["open_time"], unit="ms")

    # Append
    final_df = pd.concat([df, new_df])

    # Deduplicate just in case
    final_df.drop_duplicates(subset=["open_time"], keep="last", inplace=True)
    final_df.sort_values("open_time", inplace=True)

    # Save 1m
    final_df.to_csv(csv_path, index=False)
    logger.info(f"Updated 1m data saved to {csv_path}")

    # Resample to 5m
    resample_to_5m(final_df, symbol_clean, os.path.dirname(csv_path))


def resample_to_5m(df_1m: pd.DataFrame, symbol: str, output_dir: str) -> None:
    logger.info(f"Resampling {symbol} to 5m...")

    # Ensure datetime index
    df = df_1m.copy()
    df.set_index("date", inplace=True)

    # Resample logic
    agg_rules = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

    df_5m = df.resample("5min").agg(agg_rules).dropna()  # type: ignore

    # Reset index to keep 'date' column if needed, or match format
    df_5m["open_time"] = df_5m.index.astype(int) // 10**6  # Convert ns to ms
    df_5m["date"] = df_5m.index

    # Reorder columns
    cols = ["open_time", "open", "high", "low", "close", "volume", "date"]
    df_5m = df_5m[cols]

    # Save to disk (Standardized Name: {symbol}_5m.csv)
    output_file = os.path.join(output_dir, f"{symbol}_5m.csv")
    df_5m.to_csv(output_file, index=False)
    logger.info(f"Saved {len(df_5m)} rows to {output_file}")


if __name__ == "__main__":
    for pair in PAIRS:
        update_pair(pair)
