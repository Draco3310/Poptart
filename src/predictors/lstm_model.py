import logging

import numpy as np
import pandas as pd

from src.predictors.base_predictor import BasePredictor

logger = logging.getLogger(__name__)


class LSTMPredictor(BasePredictor):
    """
    LSTM Predictor Plugin.
    Handles 3D tensor reshaping internally.
    """

    def __init__(self, lookback_window: int = 60) -> None:
        super().__init__()
        self.lookback_window = lookback_window
        self.expected_features = None

    def load_model(self, model_path: str) -> None:
        try:
            import tensorflow as tf  # type: ignore

            self.model = tf.keras.models.load_model(model_path)

            if self.model is None:
                raise ValueError("Loaded model is None")

            # Infer expected input shape from the model
            # Input shape is usually (None, Timesteps, Features)
            input_shape = self.model.input_shape
            if input_shape and len(input_shape) == 3:
                self.lookback_window = input_shape[1]
                self.n_features = input_shape[2]
                logger.info(f"LSTM Model loaded. Expecting Input: (Batch, {self.lookback_window}, {self.n_features})")
            else:
                logger.warning(
                    f"Could not infer input shape from model: {input_shape}. "
                    f"Using default lookback: {self.lookback_window}"
                )

            logger.info(f"LSTM model loaded from {model_path}")
        except ImportError:
            logger.error("Tensorflow not installed. Cannot load LSTM model.")
            raise
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            raise

    def predict(self, enriched_df: pd.DataFrame) -> float:
        if not self.model:
            raise RuntimeError("Model not loaded.")

        # We need the last 'lookback_window' rows
        if len(enriched_df) < self.lookback_window:
            logger.warning(
                f"Not enough data for LSTM. Need {self.lookback_window}, "
                f"got {len(enriched_df)}. Returning 0.5 (Neutral)."
            )
            return 0.5

        # Select data
        # Assuming the model was trained on specific columns.
        # In a real scenario, we need strict feature alignment.
        # Here we take all numeric columns or a configured subset.
        # For simplicity, we assume the enriched_df structure matches training.
        # We drop timestamp/non-numeric columns.
        X_numeric = enriched_df.select_dtypes(include=[np.number])

        # Take last N rows
        window_data = X_numeric.iloc[-self.lookback_window :].values

        # Handle Feature Count mismatch if necessary
        if hasattr(self, "n_features") and window_data.shape[1] != self.n_features:
            # This is critical. If features don't match, we can't predict.
            # For V2 prototype, we assume config ensures alignment.
            logger.error(f"Feature count mismatch. Model expects {self.n_features}, got {window_data.shape[1]}.")
            return 0.5

        # Reshape to (1, Timesteps, Features)
        X_tensor = window_data.reshape(1, self.lookback_window, window_data.shape[1])

        prediction = self.model.predict(X_tensor, verbose=0)

        # Assuming output is a single probability [0,1] or [Bearish, Bullish]
        # If it's [Bear, Bull], we take index 1. If scalar, take it directly.
        score = float(prediction[0][0]) if prediction.ndim == 2 else float(prediction[0])

        return score
