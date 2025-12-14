import logging
from typing import Optional

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

    def predict(self, enriched_df: pd.DataFrame) -> float:
        if not self.model:
            raise RuntimeError("Model not loaded.")

        last_row = enriched_df.iloc[[-1]]

        # Feature Selection
        if self.feature_names:
            # Zero fill missing features
            missing = [f for f in self.feature_names if f not in last_row.columns]
            if missing:
                for m in missing:
                    last_row[m] = 0.0
            X = last_row[self.feature_names]
        else:
            X = last_row.select_dtypes(include=[np.number])

        # Predict Proba
        # Classes are usually [0, 1]. We want proba of class 1 (Buy).
        try:
            probs = self.model.predict_proba(X)
            # probs is [[prob_0, prob_1]]
            score = float(probs[0][1])
        except AttributeError:
            # Fallback if predict_proba not supported (e.g. Regressor)
            pred = self.model.predict(X)
            score = float(pred[0])
            # Clip to 0-1 just in case
            score = max(0.0, min(1.0, score))

        return score
