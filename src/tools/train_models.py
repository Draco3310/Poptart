import argparse
import logging
import os
import sys
from typing import Any, cast

import joblib  # type: ignore
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore

# Add project root to path
sys.path.append(os.getcwd())

from src.config import get_data_path, get_model_path
from src.core.feature_engine import FeatureEngine

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Trainer")


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
        if df["timestamp"].max() > 30000000000:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    logger.info(f"Loaded {len(df)} rows.")
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing features...")
    engine = FeatureEngine()
    enriched = engine.compute_features(df)

    # Drop rows with NaNs created by indicators
    enriched.dropna(inplace=True)
    logger.info(f"Features computed. Rows remaining: {len(enriched)}")
    return enriched


def create_target_direction(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Creates a binary target: 1 if price is higher in 'horizon' periods, 0 otherwise.
    """
    logger.info(f"Creating Directional target with horizon {horizon}...")
    df = df.copy()

    # Future Close
    df["future_close"] = df["close"].shift(-horizon)

    # Target: 1 if Future > Current, 0 otherwise
    df["target"] = (df["future_close"] > df["close"]).astype(int)

    # Drop last 'horizon' rows where target is NaN
    df.dropna(subset=["future_close"], inplace=True)

    return df


def create_target_tp_sl(df: pd.DataFrame, horizon: int, tp_mult: float = 2.5, sl_mult: float = 2.5) -> pd.DataFrame:
    """
    Creates a binary target: 1 if TP is hit before SL within 'horizon' periods.
    TP = Close + (tp_mult * ATR)
    SL = Close - (sl_mult * ATR)
    """
    logger.info(f"Creating TP/SL target with horizon {horizon}, TP={tp_mult}x, SL={sl_mult}x...")
    df = df.copy()

    # Ensure ATR exists
    if "atr" not in df.columns:
        logger.error("ATR column missing for TP/SL target creation.")
        return df

    # We need to iterate or use rolling windows. Iteration is slow but safe for complex logic.
    # Vectorized approach:
    # 1. Construct TP and SL levels for all t
    # 2. Use rolling max/min on future window? No, because levels are fixed at t.
    # We must check if High[t+k] >= TP[t] or Low[t+k] <= SL[t] for k in 1..horizon

    # For performance in this script, we'll use a simplified iteration or numpy.
    # Given dataset size (~300k rows), iteration might take a minute.

    close = df["close"].values.astype(np.float32)
    high = df["high"].values.astype(np.float32)
    low = df["low"].values.astype(np.float32)
    atr = df["atr"].values.astype(np.float32)
    n = len(df)

    targets = []
    valid_indices = []

    # Pre-calculate levels
    tp_levels = close + (tp_mult * atr)
    sl_levels = close - (sl_mult * atr)

    # Iterate (skip last 'horizon' bars)
    for i in range(n - horizon):
        tp = tp_levels[i]
        sl = sl_levels[i]

        # Look ahead window
        window_high = high[i + 1 : i + 1 + horizon]
        window_low = low[i + 1 : i + 1 + horizon]

        # Find first hit
        # np.argmax returns index of first True. If no True, returns 0 (need to check)
        tp_hits = window_high >= tp
        sl_hits = window_low <= sl

        tp_idx = np.argmax(tp_hits) if np.any(tp_hits) else horizon + 1
        sl_idx = np.argmax(sl_hits) if np.any(sl_hits) else horizon + 1

        if tp_idx < sl_idx:
            targets.append(1)  # TP hit first
        else:
            targets.append(0)  # SL hit first OR neither hit (timeout = loss)

        valid_indices.append(df.index[i])

    # Create result DF
    result = df.loc[valid_indices].copy()
    result["target"] = targets

    return result


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Any:
    logger.info("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, eval_metric="logloss", n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logger.info(f"XGBoost Accuracy: {acc:.4f}")
    return model


def train_rf(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Any:
    logger.info("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logger.info(f"Random Forest Accuracy: {acc:.4f}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TP/SL Models")
    parser.add_argument("--pair", type=str, default="XRPUSDT", help="Trading pair (e.g., XRPUSDT)")
    args = parser.parse_args()

    try:
        data_path = get_data_path(args.pair, "1m")
    except Exception as e:
        logger.error(f"Failed to get data path for {args.pair}: {e}")
        return

    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return

    # 1. Load
    df_1m = load_data(str(data_path))

    # Resample to 5m
    logger.info("Resampling to 5m...")
    agg_rules = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df = df_1m.resample("5min").agg(agg_rules).dropna()  # type: ignore

    # 2. Features
    df = prepare_features(df)

    # Define Label Configurations
    # Phase 2B: Volume Profile Enhanced Models
    LABEL_CONFIGS = [
        {"name": "tp_sl_H1", "horizon_bars": 12, "type": "tp_sl"},  # 1 Hour
    ]

    for cfg in LABEL_CONFIGS:
        logger.info(f"--- Processing Label Config: {cfg['name']} ---")

        # 3. Create Target
        if cfg["type"] == "direction":
            df_labeled = create_target_direction(df, horizon=int(cast(int, cfg["horizon_bars"])))
        elif cfg["type"] == "tp_sl":
            df_labeled = create_target_tp_sl(df, horizon=int(cast(int, cfg["horizon_bars"])), tp_mult=2.5, sl_mult=2.5)
        else:
            logger.warning(f"Unknown label type: {cfg['type']}")
            continue

        # 4. Split
        # Features to use (exclude target and future columns)
        feature_cols = [c for c in df_labeled.columns if c not in ["target", "future_close", "trades"]]

        X = df_labeled[feature_cols]
        y = df_labeled["target"]

        # Log Class Balance
        pos_rate = y.mean()
        logger.info(f"Class Balance (Positive Rate): {pos_rate:.4f}")

        # Time-based split (no shuffle)
        split_idx = int(len(df_labeled) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

        # Check for new features
        new_feats = [f for f in feature_cols if "poc" in f or "vah" in f or "val" in f]
        logger.info(f"Volume Profile Features detected: {new_feats}")

        # 5. Train & Save
        xgb_path = get_model_path(args.pair, f"xgb_{cfg['name']}", ext=".model")
        rf_path = get_model_path(args.pair, f"rf_{cfg['name']}", ext=".joblib")

        xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
        xgb_model.save_model(xgb_path)
        logger.info(f"Saved {xgb_path}")

        rf_model = train_rf(X_train, y_train, X_test, y_test)
        joblib.dump(rf_model, rf_path)
        logger.info(f"Saved {rf_path}")

    logger.info("All training tasks complete.")


if __name__ == "__main__":
    main()
