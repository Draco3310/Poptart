import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.predictors.base_predictor import BasePredictor

logger = logging.getLogger(__name__)


class RandomForestPredictor(BasePredictor):
    """
    Random Forest Predictor Plugin (Scikit-Learn).
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_names: Optional[list[str]] = None

    def load_model(self, model_path: str) -> None:
        try:
            import joblib  # type: ignore

            self.model = joblib.load(model_path)
            logger.info(f"Random Forest model loaded from {model_path}")

            if self.model is None:
                raise ValueError("Loaded model is None")

            # Check for feature names if available
            if hasattr(self.model, "feature_names_in_"):
                self.feature_names = list(self.model.feature_names_in_)
            else:
                self.feature_names = None

        except ImportError:
            logger.error("Joblib not installed. Cannot load Random Forest model.")
            raise
        except Exception as e:
            logger.error(f"Failed to load Random Forest model: {e}")
            raise

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to select and align features."""
        if self.feature_names:
            # Zero fill missing features
            missing = [f for f in self.feature_names if f not in df.columns]
            if missing:
                new_cols = {m: 0.0 for m in missing}
                df = df.assign(**new_cols)
            return df[self.feature_names]
        else:
            return df.select_dtypes(include=[np.number])

    def predict_single(self, enriched_df: pd.DataFrame) -> float:
        if not self.model:
            raise RuntimeError("Model not loaded.")

        # Take last row
        target_df = enriched_df.iloc[[-1]]
        X = self._prepare_features(target_df)

        try:
            probs = self.model.predict_proba(X)
            return float(probs[0][1])
        except AttributeError:
            pred = self.model.predict(X)
            score = float(pred[0])
            return max(0.0, min(1.0, score))

    def predict_batch(self, enriched_df: pd.DataFrame) -> Any:
        if not self.model:
            raise RuntimeError("Model not loaded.")

        X = self._prepare_features(enriched_df)

        try:
            probs = self.model.predict_proba(X)
            return probs[:, 1]
        except AttributeError:
            return self.model.predict(X)
