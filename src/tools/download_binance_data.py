import io
import logging
import os
import sys
import zipfile

import pandas as pd
import requests
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config import SYMBOL_MAP

# Load environment variables
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("BinanceDownloader")

# Configuration
SYMBOLS = ["BTCUSDT", "XRPUSDT", "SOLUSDT"]
YEARS = ["2024", "2025"]
MONTHS = [f"{i:02d}" for i in range(1, 13)]  # 01 to 12
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
INTERVAL = "1m"


def download_and_merge(symbol: str) -> None:
    print(f"Downloading {symbol} data...")
    all_dfs = []

    coin_dir = SYMBOL_MAP.get(symbol, symbol)
    output_dir = os.path.join("data", str(coin_dir))
    os.makedirs(output_dir, exist_ok=True)

    # Configure proxies if present
    proxies: dict[str, str] = {}
    http_proxy = os.getenv("HTTP_PROXY")
    if http_proxy:
        proxies["http"] = http_proxy
    https_proxy = os.getenv("HTTPS_PROXY")
    if https_proxy:
        proxies["https"] = https_proxy

    for year in YEARS:
        for month in MONTHS:
            # Construct the URL for the monthly zip
            file_name = f"{symbol}-{INTERVAL}-{year}-{month}.zip"
            url = f"{BASE_URL}/{symbol}/{INTERVAL}/{file_name}"

            try:
                # print(f"Downloading {year}-{month}...", end=" ", flush=True)
                response = requests.get(url, proxies=proxies)

                if response.status_code == 404:
                    # print("Not found (likely not finished yet).")
                    continue

                # Unzip in memory
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    # The zip usually contains one csv file
                    csv_name = z.namelist()[0]
                    with z.open(csv_name) as f:
                        # Binance data has no headers. Define them manually.
                        df = pd.read_csv(f, header=None)
                        df.columns = [
                            "open_time",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "close_time",
                            "quote_volume",
                            "count",
                            "taker_buy_volume",
                            "taker_buy_quote_volume",
                            "ignore",
                        ]

                        # Keep only essential columns for backtesting
                        df = df[["open_time", "open", "high", "low", "close", "volume"]]
                        all_dfs.append(df)
                        # print("Success.")

            except Exception as e:
                print(f"Error downloading {year}-{month}: {e}")

    if all_dfs:
        # Merge all months
        final_df = pd.concat(all_dfs)

        # Normalize timestamps (Handle potential microsecond data)
        # If timestamp > 3e12 (Year 2065), assume microseconds and convert to ms
        mask_us = final_df["open_time"] > 3000000000000
        if mask_us.any():
            logger.warning(f"Detected {mask_us.sum()} rows with microsecond timestamps. Converting to ms.")
            final_df.loc[mask_us, "open_time"] = final_df.loc[mask_us, "open_time"] // 1000

        # specific cleanup
        final_df["date"] = pd.to_datetime(final_df["open_time"], unit="ms")
        final_df.sort_values("open_time", inplace=True)

        # Save to disk (Standardized Name: {symbol}_1m.csv)
        output_file = os.path.join(output_dir, f"{symbol}_1m.csv")
        final_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(final_df)} rows to {output_file}")

        # Resample to 5m for XRP and SOL
        if symbol in ["XRPUSDT", "SOLUSDT"]:
            resample_to_5m(final_df, symbol, output_dir)

        print(f"Completed {symbol}.")

    else:
        logger.warning(f"No data found for {symbol}")


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
    # Run for all symbols
    for sym in SYMBOLS:
        download_and_merge(sym)
