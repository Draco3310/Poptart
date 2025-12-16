import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.config import Config
from src.core.features.base import FeatureBlock

logger = logging.getLogger(__name__)


class BetaFeatureBlock(FeatureBlock):
    """
    Calculates High-Beta features relative to a benchmark (BTC).
    Includes Rolling Beta, Relative Strength, and Correlation.
    """

    def apply(self, df: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        if df.empty:
            return df

        # Check if BTC context is available
        if not context or "btc_df" not in context:
            # Fill with 0 or NaN to avoid breaking models expecting these columns
            df["beta_24h"] = 0.0
            df["rs_ratio"] = 1.0
            df["btc_corr_4h"] = 0.0
            return df

        btc_df = context["btc_df"]
        if btc_df.empty:
            return df

        df = df.copy()

        # Align DataFrames on Timestamp
        # We assume both are 1m candles.

        # Merge
        # We use 'left' join to keep SOL timestamps
        merged = df[["close"]].merge(
            btc_df[["close"]], left_index=True, right_index=True, how="left", suffixes=("", "_btc")
        )

        # Forward fill BTC data (if SOL has candles where BTC doesn't, e.g. maintenance)
        merged["close_btc"] = merged["close_btc"].ffill()

        # Calculate Returns
        merged["ret_sol"] = merged["close"].pct_change()
        merged["ret_btc"] = merged["close_btc"].pct_change()

        # Determine window size dynamically based on data frequency
        if len(df) > 1:
            # Calculate minutes per candle
            time_diff = df.index[1] - df.index[0]
            minutes = time_diff.total_seconds() / 60
            if minutes <= 0:
                minutes = 5  # Fallback to avoid div/0

            window_24h = int(1440 / minutes)
            window_4h = int(240 / minutes)
        else:
            # Fallback if not enough data to infer
            # Default to 5m assumption (288 bars for 24h)
            window_24h = 288
            window_4h = 48

        # 1. Rolling Beta (24h)
        # Beta = Cov(SOL, BTC) / Var(BTC)
        cov = merged["ret_sol"].rolling(window=window_24h).cov(merged["ret_btc"])
        var = merged["ret_btc"].rolling(window=window_24h).var()
        merged["beta_24h"] = cov / var.replace(0, np.nan)

        # 2. Relative Strength (RS) Ratio
        # RS = SOL / BTC
        merged["rs_ratio"] = merged["close"] / merged["close_btc"].replace(0, np.nan)

        # 3. Rolling Correlation (4h)
        merged["btc_corr_4h"] = merged["ret_sol"].rolling(window=window_4h).corr(merged["ret_btc"])

        # Assign back to df
        df["beta_24h"] = merged["beta_24h"]
        df["rs_ratio"] = merged["rs_ratio"]
        df["btc_corr_4h"] = merged["btc_corr_4h"]

        return df
