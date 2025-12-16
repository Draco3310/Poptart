from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.config import Config
from src.core.features.base import FeatureBlock


class TransformedFeatureBlock(FeatureBlock):
    """
    Adds transformed technical indicators to ensure stationarity.
    Includes Z-Scores, Distance-to-MA, and Volatility Ratios.
    """

    def apply(self, df: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()

        # 1. Rolling Z-Score of Close (1H window)
        # Z = (Price - Mean) / Std
        # Use time-based rolling to support any timeframe
        window = '1h'
        rolling_mean = df["close"].rolling(window=window).mean()
        rolling_std = df["close"].rolling(window=window).std()
        df["zscore_close_1h"] = (df["close"] - rolling_mean) / rolling_std.replace(0, np.nan)

        # 2. Distance to EMA 200 (Trend Extension)
        # (Price - EMA) / Price
        if "ema200" not in df.columns:
            ema_slow = 200
            if context and "pair_config" in context:
                ema_slow = getattr(context["pair_config"], "ema_period_slow", 200)
            df["ema200"] = ta.ema(df["close"], length=ema_slow)

        df["dist_ema200"] = (df["close"] - df["ema200"]) / df["close"]

        # 3. Log Returns (Stationary Price Change)
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        return df
