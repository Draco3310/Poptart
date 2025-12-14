from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from numba import jit  # type: ignore

from src.core.features.base import FeatureBlock


@jit(nopython=True)
def calculate_hurst_numba(series: np.ndarray) -> float:
    """
    Calculates the Hurst Exponent of a time series using the R/S analysis method.
    Optimized with Numba for performance.
    """
    if len(series) < 20:
        return 0.5

    # 1. Calculate R/S for the full chunk
    # Mean centered series
    mean = np.mean(series)
    centered = series - mean

    # Cumulative deviation
    z = np.cumsum(centered)

    # Range
    r = np.max(z) - np.min(z)

    # Standard deviation
    s = np.std(series)  # ddof=1 not supported in all numba versions for std?
    # Numba np.std usually supports ddof. Let's check.
    # If not, we can calculate it manually or use ddof=0 (population) which is close enough for N=100.
    # Actually, let's use manual calculation to be safe and fast.

    # Manual std with ddof=1
    var = np.sum((series - mean) ** 2) / (len(series) - 1)
    s = np.sqrt(var)

    if s == 0:
        return 0.5

    rs = r / s

    # Hurst estimate: RS ~ (N/2)^H  => log(RS) ~ H * log(N/2)
    h = np.log(rs) / np.log(len(series) / 2)

    return float(h)


class StatisticalFeatureBlock(FeatureBlock):
    """
    Adds statistical features to detect market regimes (Trend vs Mean Reversion).
    Includes Hurst Exponent and Autocorrelation.
    """

    def apply(self, df: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()

        # 1. Rolling Autocorrelation (Lag 1)
        # Positive = Trend, Negative = Mean Reversion
        window = 20
        # Optimized: Use rolling correlation with shifted series instead of .apply()
        df["autocorr_20"] = df["close"].rolling(window=window).corr(df["close"].shift(1))

        # 2. Hurst Exponent (Rolling)
        # H < 0.5 = Mean Reversion, H > 0.5 = Trend
        # We use a rolling window to calculate the Hurst exponent.
        # To maintain performance, we use a simplified R/S analysis or a smaller window.
        # Here we implement a rolling calculation using the standard deviation of differences (simplified).

        # Note: Real R/S analysis is O(N^2) or O(N log N). For a rolling window of 100 on 5m data:
        # We can use a helper function.

        hurst_window = 100
        # Use Numba-optimized function
        # engine='numba', engine_kwargs={'nopython': True, 'nogil': True} is supported in recent pandas
        # But .apply(func, raw=True) with a numba-jitted func is also fast.
        df["hurst_exponent"] = df["close"].rolling(window=hurst_window).apply(calculate_hurst_numba, raw=True)

        # Keep Efficiency Ratio as a faster, complementary metric
        change = df["close"].diff().abs()
        net_change = (df["close"] - df["close"].shift(window)).abs()
        sum_change = change.rolling(window=window).sum()
        df["efficiency_ratio"] = net_change / sum_change.replace(0, np.nan)

        return df
