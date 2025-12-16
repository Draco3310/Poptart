import argparse
import logging
import os
import sys

import joblib  # type: ignore
import pandas as pd
import xgboost as xgb

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config import get_data_path, get_model_path
from src.core.feature_engine import FeatureEngine
from src.utils.data_loader import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_scores(pair: str, version: str) -> None:
    try:
        data_path = get_data_path(pair, "1m")
        rf_path = get_model_path(pair, f"rf_tp_sl_H1_{version}", ext=".joblib")
        xgb_path = get_model_path(pair, f"xgb_tp_sl_H1_{version}", ext=".model")
    except Exception as e:
        logger.error(f"Failed to resolve paths: {e}")
        return

    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return

    # 1. Load Data
    df = load_data(str(data_path))

    # Resample to 5m
    agg_rules = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df_5m = df.resample("5min").agg(agg_rules).dropna()  # type: ignore

    # 2. Compute Features
    logger.info("Computing features...")
    feature_engine = FeatureEngine()
    df_5m = feature_engine.compute_features(df_5m)

    # Drop NaNs
    df_5m.dropna(inplace=True)

    # 3. Load Models
    logger.info("Loading models...")
    if not os.path.exists(rf_path) or not os.path.exists(xgb_path):
        logger.error(f"Model files not found: {rf_path} or {xgb_path}")
        return

    rf_model = joblib.load(rf_path)
    xgb_model = xgb.Booster()
    xgb_model.load_model(xgb_path)

    # 4. Predict
    # Prepare features (ensure columns match training)
    # We need to know which columns were used.
    # Usually FeatureEngine output minus target/future.
    # For now, we'll use all numeric columns that are not targets.
    # Ideally, we should save the feature list with the model.
    # Based on train_models.py:
    # feature_cols = [c for c in df_labeled.columns if c not in ["target", "future_close", "trades"]]

    feature_cols = [c for c in df_5m.columns if c not in ["target", "future_close", "trades"]]
    X = df_5m[feature_cols]

    logger.info(f"Predicting on {len(X)} rows...")

    # RF
    rf_probs = rf_model.predict_proba(X)[:, 1]

    # XGB
    dmatrix = xgb.DMatrix(X)
    xgb_probs = xgb_model.predict(dmatrix)

    # Ensemble (Average)
    ensemble_probs = (rf_probs + xgb_probs) / 2.0

    # 5. Analyze Distribution
    logger.info("\n--- Score Distribution ---")
    stats = pd.Series(ensemble_probs).describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    print(stats)

    # Check potential thresholds
    print("\n--- Threshold Analysis ---")
    print(f"Rows > 0.60: {(ensemble_probs > 0.60).sum()} ({(ensemble_probs > 0.60).mean():.2%})")
    print(f"Rows > 0.65: {(ensemble_probs > 0.65).sum()} ({(ensemble_probs > 0.65).mean():.2%})")
    print(f"Rows > 0.70: {(ensemble_probs > 0.70).sum()} ({(ensemble_probs > 0.70).mean():.2%})")

    print(f"Rows < 0.40: {(ensemble_probs < 0.40).sum()} ({(ensemble_probs < 0.40).mean():.2%})")
    print(f"Rows < 0.35: {(ensemble_probs < 0.35).sum()} ({(ensemble_probs < 0.35).mean():.2%})")
    print(f"Rows < 0.30: {(ensemble_probs < 0.30).sum()} ({(ensemble_probs < 0.30).mean():.2%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Model Scores")
    parser.add_argument("--pair", type=str, default="XRPUSDT", help="Trading pair")
    parser.add_argument("--version", type=str, default="microvol_v1", help="Model version tag")
    args = parser.parse_args()

    analyze_scores(args.pair, args.version)
