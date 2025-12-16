import abc
import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class BasePredictor(abc.ABC):
    """
    Abstract Base Class for ML Predictors.
    """

    def __init__(self) -> None:
        self.model: Optional[Any] = None

    @abc.abstractmethod
    def load_model(self, model_path: str) -> None:
        """
        Loads the model from the specified path.
        """
        pass

    @abc.abstractmethod
    def predict_single(self, enriched_df: pd.DataFrame) -> float:
        """
        Generates a prediction score (0.0 to 1.0) for the *last* row of data.

        Args:
            enriched_df: DataFrame containing all computed features.

        Returns:
            float: Confidence score between 0.0 (Strong Sell/Bearish) and 1.0 (Strong Buy/Bullish).
        """
        pass

    def predict_batch(self, enriched_df: pd.DataFrame) -> Any:
        """
        Generates prediction scores for the entire DataFrame.

        Args:
            enriched_df: DataFrame containing all computed features.

        Returns:
            np.ndarray or pd.Series: Array of confidence scores.
        """
        raise NotImplementedError("Batch prediction not implemented for this predictor.")

    def predict(self, enriched_df: pd.DataFrame) -> Any:
        """
        DEPRECATED: Use predict_single or predict_batch instead.
        Wrapper that attempts to dispatch based on input size.
        """
        # Heuristic: If large dataframe, assume batch.
        # Note: This is fragile for models like LSTM that need large lookback for single inference.
        if len(enriched_df) > 200:
            try:
                return self.predict_batch(enriched_df)
            except NotImplementedError:
                pass
        
        return self.predict_single(enriched_df)
