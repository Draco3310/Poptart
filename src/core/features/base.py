from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd


class FeatureBlock(ABC):
    """
    Abstract base class for feature engineering blocks.
    Each block is responsible for adding a specific set of features to the DataFrame.
    """

    @abstractmethod
    def apply(self, df: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Applies the feature transformation to the DataFrame.

        Args:
            df: The input DataFrame (OHLCV + existing features).
            context: Optional dictionary containing external data (e.g., 'btc_df').

        Returns:
            The DataFrame with new features added.
        """
        pass
