import logging
import os
import sys
from typing import Any

import pandas as pd

# Add project root to path
sys.path.append(os.getcwd())

from src.core.feature_engine import FeatureEngine

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RegimeLabeler")


def load_data(filepath: str) -> pd.DataFrame:
    logger.info(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, header=None, names=["timestamp", "open", "high", "low", "close", "volume", "trades"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df


def label_regimes(df: pd.DataFrame, method: str = "adx", **kwargs: Any) -> pd.DataFrame:
    """
    Labels regimes based on Volatility and Trend Strength.
    0: RANGE
    1: TREND
    2: CRASH

    Args:
        df: DataFrame with features
        method: 'adx' (Legacy) or 'future_returns' (Predictive)
        kwargs: Thresholds (adx_threshold, vol_crash_threshold, trend_return_threshold, horizon_candles)
    """
    df = df.copy()

    # Ensure features exist
    if "adx" not in df.columns or "atr" not in df.columns:
        logger.info("Computing features for labeling...")
        engine = FeatureEngine()
        df = engine.compute_features(df)

    # 1. Volatility Metric
    df["volatility"] = df["atr"] / df["close"]

    # 2. Trend Metric
    df["ema_bias"] = (df["close"] - df["ema200"]) / df["ema200"]

    # Thresholds
    VOL_CRASH_THRESHOLD = kwargs.get("vol_crash_threshold", 0.005)

    # Initialize labels
    df["regime_label"] = 0  # Default to RANGE

    if method == "adx":
        # Legacy Method: Based on current ADX
        ADX_TREND_THRESHOLD = kwargs.get("adx_threshold", 25)

        # Label TREND
        df.loc[df["adx"] > ADX_TREND_THRESHOLD, "regime_label"] = 1

    elif method == "future_returns":
        # Predictive Method: Based on future price movement
        horizon = kwargs.get("horizon_candles", 48)  # Default 4h (48 * 5m)
        threshold = kwargs.get("trend_return_threshold", 0.02)  # Default 2%

        # Calculate future returns (absolute)
        # We look at the max excursion or just the close?
        # "Price moves > 2%" usually implies directional move.
        # Let's use absolute return over horizon.
        future_close = df["close"].shift(-horizon)
        df["future_return"] = (future_close - df["close"]) / df["close"]

        # Label TREND if absolute return > threshold
        df.loc[df["future_return"].abs() > threshold, "regime_label"] = 1

        # Remove last 'horizon' rows as they have NaN targets
        df = df.dropna(subset=["future_return"])

    else:
        raise ValueError(f"Unknown labeling method: {method}")

    # Label CRASH (Overrides TREND)
    # High Volatility is always a CRASH regime regardless of method
    df.loc[df["volatility"] > VOL_CRASH_THRESHOLD, "regime_label"] = 2

    return df


def analyze_distribution(df: pd.DataFrame) -> None:
    counts = df["regime_label"].value_counts().sort_index()
    total = len(df)

    logger.info("Regime Distribution:")
    logger.info(f"0 (RANGE): {counts.get(0, 0)} ({counts.get(0, 0) / total:.1%})")
    logger.info(f"1 (TREND): {counts.get(1, 0)} ({counts.get(1, 0) / total:.1%})")
    logger.info(f"2 (CRASH): {counts.get(2, 0)} ({counts.get(2, 0) / total:.1%})")


if __name__ == "__main__":
    data_path = "data/XRPUSDT_1.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    df_1m = load_data(data_path)

    # Resample to 5m
    agg_rules = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df_5m = df_1m.resample("5min").agg(agg_rules).dropna()  # type: ignore

    df_labeled = label_regimes(df_5m)
    analyze_distribution(df_labeled)
