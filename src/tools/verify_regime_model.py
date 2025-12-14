import logging
import os
import sys
from collections import Counter

import pandas as pd

# Add project root to path
sys.path.append(os.getcwd())

from src.core.feature_engine import FeatureEngine
from src.predictors.regime_classifier import RegimeClassifier

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RegimeVerifier")


def load_data(filepath: str) -> pd.DataFrame:
    logger.info(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, header=None, names=["timestamp", "open", "high", "low", "close", "volume", "trades"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df


def verify_model() -> None:
    data_path = "data/XRPUSDT_1.csv"

    # 1. Load & Resample
    df_1m = load_data(data_path)
    agg_rules = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df = df_1m.resample("5min").agg(agg_rules).dropna()  # type: ignore

    # 2. Compute Features
    logger.info("Computing features...")
    engine = FeatureEngine()
    df = engine.compute_features(df)

    # Add derived features manually for verification
    df["volatility"] = df["atr"] / df["close"]
    df["ema_bias"] = (df["close"] - df["ema200"]) / df["ema200"]

    # 3. Load Classifier
    classifier = RegimeClassifier()

    # 4. Predict
    logger.info("Predicting...")

    # Batch prediction for speed (simulating what happens in backtest loop but vectorized if possible)
    # The classifier.predict method takes a DataFrame and uses the last row.
    # But the underlying model can take a DataFrame of many rows.

    if classifier.model:
        # Use underlying model directly for batch
        X = df[classifier.features]
        raw_preds = classifier.model.predict(X)

        counts = Counter(raw_preds)
        logger.info(f"Distribution: {counts}")

        # Check mapping
        # 0: RANGE, 1: TREND, 2: CRASH
        total = sum(counts.values())
        logger.info(f"RANGE: {counts[0] / total:.1%}")
        logger.info(f"TREND: {counts[1] / total:.1%}")
        logger.info(f"CRASH: {counts[2] / total:.1%}")
    else:
        logger.error("Model not loaded")


if __name__ == "__main__":
    verify_model()
