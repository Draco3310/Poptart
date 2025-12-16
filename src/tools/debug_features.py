import argparse
import logging
import os
import sys

import pandas as pd

# Add project root to path
sys.path.append(os.getcwd())

from src.config import get_data_path
from src.core.feature_engine import FeatureEngine
from src.utils.data_loader import load_data

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DebugFeatures")


def debug_features(pair: str) -> None:
    # Load Data
    try:
        data_path = str(get_data_path(pair, "1m"))
    except Exception as e:
        logger.error(f"Failed to get data path: {e}")
        return

    if not os.path.exists(data_path):
        logger.error(f"Data not found: {data_path}")
        return

    logger.info(f"Loading data from {data_path}...")
    df = load_data(data_path)

    # Limit to 1000 rows for speed
    df = df.head(1000)

    fe = FeatureEngine()
    logger.info("Computing features...")
    enriched = fe.compute_features(df)

    logger.info("\n--- Columns ---")
    logger.info(enriched.columns.tolist())

    logger.info("\n--- Head (Last 5) ---")
    logger.info(enriched[["close", "rsi", "adx", "atr", "ema200"]].tail())

    logger.info("\n--- RSI Stats ---")
    logger.info(enriched["rsi"].describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Features")
    parser.add_argument("--pair", type=str, default="SOLUSDT", help="Trading pair")
    args = parser.parse_args()

    debug_features(args.pair)
