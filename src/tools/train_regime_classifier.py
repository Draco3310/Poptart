import argparse
import logging
import os
import sys

import joblib  # type: ignore
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import accuracy_score, classification_report  # type: ignore

# Add project root to path
sys.path.append(os.getcwd())

from src.config import get_data_path, get_model_path, get_pair_config
from src.core.feature_engine import FeatureEngine
from src.tools.label_regimes import label_regimes

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RegimeTrainer")


def load_data(filepath: str) -> pd.DataFrame:
    logger.info(f"Loading data from {filepath}...")

    # Check for header
    peek = pd.read_csv(filepath, nrows=1, header=None)
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
        df = pd.read_csv(filepath, header=None, names=["timestamp", "open", "high", "low", "close", "volume", "trades"])

    # Downcast numeric columns to float32
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    # Convert timestamp
    # Check if it's already datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        # Check if it's ms or s
        # If max > 3e10 (year 2920), it's probably ms
        if df["timestamp"].max() > 30000000000:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        elif df["timestamp"].max() > 3000000000:  # Year 2065 in seconds
            # Could be ms if data is very recent, but usually s
            # Let's assume ms if it's huge, s if it's reasonable timestamp
            # But wait, 1700000000 is 2023 in seconds.
            # 1700000000000 is 2023 in ms.
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        else:
            # Try auto
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df


def train_regime_classifier(
    pair: str, method: str, threshold: float, horizon: int, version: str, vol_threshold: float = 0.005
) -> None:
    # 1. Get Config & Data Path
    try:
        get_pair_config(pair)
        data_path = get_data_path(pair, "1m")
    except Exception as e:
        logger.error(f"Failed to get config for {pair}: {e}")
        return

    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return

    # 2. Load & Resample
    df_1m = load_data(str(data_path))
    agg_rules = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df = df_1m.resample("5min").agg(agg_rules).dropna()  # type: ignore

    # 3. Compute Features (Input for Model)
    logger.info("Computing features...")
    engine = FeatureEngine()
    df = engine.compute_features(df)

    # 4. Label Regimes (Target for Model)
    logger.info(f"Labeling regimes using method='{method}' (Threshold={threshold}, Horizon={horizon})...")

    label_kwargs = {}
    label_kwargs["vol_crash_threshold"] = vol_threshold

    if method == "adx":
        label_kwargs["adx_threshold"] = threshold
    elif method == "future_returns":
        label_kwargs["trend_return_threshold"] = threshold
        label_kwargs["horizon_candles"] = horizon

    df = label_regimes(df, method=method, **label_kwargs)

    # Analyze Distribution
    counts = df["regime_label"].value_counts().sort_index()
    total = len(df)
    logger.info("Regime Distribution:")
    logger.info(f"0 (RANGE): {counts.get(0, 0)} ({counts.get(0, 0) / total:.1%})")
    logger.info(f"1 (TREND): {counts.get(1, 0)} ({counts.get(1, 0) / total:.1%})")
    logger.info(f"2 (CRASH): {counts.get(2, 0)} ({counts.get(2, 0) / total:.1%})")

    # 5. Prepare Dataset
    feature_cols = [
        c
        for c in df.columns
        if c not in ["regime_label", "open", "high", "low", "close", "volume", "trades", "timestamp", "future_return"]
    ]

    # Drop NaNs
    df = df.dropna()

    X = df[feature_cols]
    y = df["regime_label"]

    # 6. Split
    # Time-based split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

    # 7. Train
    logger.info("Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, class_weight="balanced", n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # 8. Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info("\n" + str(classification_report(y_test, preds, target_names=["RANGE", "TREND", "CRASH"])))

    # 9. Save
    model_path = get_model_path(pair, "rf_regime", version=version, ext=".joblib")

    # Save model and feature list together
    artifact = {"model": clf, "features": feature_cols}
    joblib.dump(artifact, model_path)
    logger.info(f"Saved model and features to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Regime Classifier")
    parser.add_argument("--pair", type=str, default="XRPUSDT", help="Trading pair (e.g., XRPUSDT)")
    parser.add_argument("--method", type=str, default="adx", choices=["adx", "future_returns"], help="Labeling method")
    parser.add_argument("--threshold", type=float, default=25, help="Threshold for labeling (ADX value or Return %)")
    parser.add_argument(
        "--horizon", type=int, default=48, help="Horizon in candles for future returns (default 48 = 4h)"
    )
    parser.add_argument("--version", type=str, default="v1", help="Version tag for the model file")
    parser.add_argument("--vol_threshold", type=float, default=0.005, help="Volatility threshold for CRASH regime")

    args = parser.parse_args()

    train_regime_classifier(
        args.pair, args.method, args.threshold, args.horizon, args.version, vol_threshold=args.vol_threshold
    )
