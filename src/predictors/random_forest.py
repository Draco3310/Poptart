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

    def predict(self, enriched_df: pd.DataFrame) -> Any:
        if not self.model:
            raise RuntimeError("Model not loaded.")

        target_df = enriched_df
        is_batch = len(enriched_df) > 200

        if not is_batch:
            target_df = enriched_df.iloc[[-1]]

        # Feature Selection
        if self.feature_names:
            # Zero fill missing features
            missing = [f for f in self.feature_names if f not in target_df.columns]
            if missing:
                # Use assign to avoid SettingWithCopyWarning
                for m in missing:
                    target_df = target_df.assign(**{m: 0.0})
            X = target_df[self.feature_names]
        else:
            X = target_df.select_dtypes(include=[np.number])

        # Predict Proba
        # Classes are usually [0, 1]. We want proba of class 1 (Buy).
        try:
            probs = self.model.predict_proba(X)
            # probs is [[prob_0, prob_1], [prob_0, prob_1], ...]
            if is_batch:
                return probs[:, 1] # Return array of class 1 probabilities
            else:
                score = float(probs[0][1])
                return score
        except AttributeError:
            # Fallback if predict_proba not supported (e.g. Regressor)
            pred = self.model.predict(X)
            if is_batch:
                return pred
            else:
                score = float(pred[0])
                # Clip to 0-1 just in case
                score = max(0.0, min(1.0, score))
                return score
