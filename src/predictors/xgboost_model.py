import logging
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from src.predictors.base_predictor import BasePredictor

logger = logging.getLogger(__name__)


class XGBoostPredictor(BasePredictor):
    """
    XGBoost Predictor Plugin.
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_names: Optional[List[str]] = []

    def load_model(self, model_path: str) -> None:
        """
        Loads the XGBoost model.
        """
        try:
            import xgboost as xgb

            self.model = xgb.Booster()
            if self.model is None:
                raise ValueError("Failed to initialize XGBoost Booster")

            self.model.load_model(model_path)

            # Try to infer feature names if available, or expect them in a separate config
            # For this V2 implementation, we'll assume the model file preserves feature names
            # or we might need a sidecar config.
            # If the model was saved with feature names, `feature_names` property might work.
            try:
                self.feature_names = list(self.model.feature_names) if self.model.feature_names else None
            except Exception:
                logger.warning(
                    "Could not extract feature names from XGBoost model. Prediction might fail if column order differs."
                )

            logger.info(f"XGBoost model loaded from {model_path}")
        except ImportError:
            logger.error("XGBoost not installed. Cannot load model.")
            raise
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            raise

    def predict(self, enriched_df: pd.DataFrame) -> Any:
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            import xgboost as xgb
        except ImportError:
            raise RuntimeError("XGBoost not installed.")

        # Determine if Batch or Single Row
        # If called from TradingAgent, it passes a slice (usually 50 rows), but we only want the last one.
        # If called from Backtest Batch, we want all rows.
        # Heuristic: If > 100 rows, assume Batch?
        # Or better: The caller should handle slicing.
        # But TradingAgent passes a lookback window (50 rows).
        # So for TradingAgent, we MUST take the last row.
        # For Batch Backtest, we pass 500k rows.
        
        target_df = enriched_df
        is_batch = len(enriched_df) > 200 # Arbitrary threshold to distinguish lookback from full history

        if not is_batch:
            target_df = enriched_df.iloc[[-1]]

        # Ensure we select only the features the model expects, if known
        if self.feature_names:
            # Check if all features exist
            missing = [f for f in self.feature_names if f not in target_df.columns]
            if missing:
                logger.warning(f"Missing features for XGBoost: {missing}. Filling with 0.")
                # Use assign to avoid SettingWithCopyWarning on slice
                for m in missing:
                    target_df = target_df.assign(**{m: 0.0})

            X = target_df[self.feature_names]
        else:
            # If no feature names known, use all numeric columns
            X = target_df.select_dtypes(include=[np.number])

        dtest = xgb.DMatrix(X)
        prediction = self.model.predict(dtest)

        if is_batch:
            return prediction # Returns numpy array
        else:
            # Prediction is typically a numpy array
            score = float(prediction[0])
            return score
