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
    def predict(self, enriched_df: pd.DataFrame) -> float:
        """
        Generates a prediction score (0.0 to 1.0) for the *last* row of data.

        Args:
            enriched_df: DataFrame containing all computed features.

        Returns:
            float: Confidence score between 0.0 (Strong Sell/Bearish) and 1.0 (Strong Buy/Bullish).
        """
        pass
