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

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to select and align features."""
        if self.feature_names:
            missing = [f for f in self.feature_names if f not in df.columns]
            if missing:
                logger.warning(f"Missing features for XGBoost: {missing}. Filling with 0.")
                new_cols = {m: 0.0 for m in missing}
                df = df.assign(**new_cols)
            return df[self.feature_names]
        else:
            return df.select_dtypes(include=[np.number])

    def predict_single(self, enriched_df: pd.DataFrame) -> float:
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import xgboost as xgb

        # Take last row
        target_df = enriched_df.iloc[[-1]]
        X = self._prepare_features(target_df)

        dtest = xgb.DMatrix(X)
        prediction = self.model.predict(dtest)
        return float(prediction[0])

    def predict_batch(self, enriched_df: pd.DataFrame) -> Any:
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import xgboost as xgb

        X = self._prepare_features(enriched_df)
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
