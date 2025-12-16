import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from numba import jit  # type: ignore

from src.core.features.base import FeatureBlock

logger = logging.getLogger(__name__)


@jit(nopython=True)
def calculate_value_area_numba(closes: np.ndarray, volumes: np.ndarray, bins: int) -> Tuple[float, float, float]:
    """
    Calculates POC, VAH, and VAL using a histogram approach.
    Optimized with Numba for performance.
    """
    min_p = np.min(closes)
    max_p = np.max(closes)

    if min_p == max_p:
        return min_p, max_p, min_p

    # Manual Histogram
    hist = np.zeros(bins, dtype=np.float64)
    bin_width = (max_p - min_p) / bins

    for i in range(len(closes)):
        bin_idx = int((closes[i] - min_p) / bin_width)
        if bin_idx >= bins:
            bin_idx = bins - 1
        hist[bin_idx] += volumes[i]

    # POC
    max_idx = np.argmax(hist)
    poc = min_p + (max_idx + 0.5) * bin_width

    # Value Area Expansion
    total_vol = np.sum(hist)
    target_vol = total_vol * 0.70
    current_vol = hist[max_idx]
    left = max_idx
    right = max_idx

    while current_vol < target_vol:
        vol_left = 0.0 if left == 0 else hist[left - 1]
        vol_right = 0.0 if right == bins - 1 else hist[right + 1]

        if vol_left == 0 and vol_right == 0:
            break

        if vol_left > vol_right:
            current_vol += vol_left
            left -= 1
        else:
            current_vol += vol_right
            right += 1

    val = min_p + left * bin_width
    vah = min_p + (right + 1) * bin_width

    return poc, vah, val


class VolumeProfileFeatureBlock(FeatureBlock):
    """
    Computes Rolling Volume Profile features.
    - POC (Point of Control): Price level with highest volume.
    - VAH (Value Area High): Highest price in the 70% volume area.
    - VAL (Value Area Low): Lowest price in the 70% volume area.
    """

    def __init__(self, window_hours: int = 24, bins: int = 100):
        self.window_hours = window_hours
        self.bins = bins

    def apply(self, df: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Applies volume profile calculation.
        1. Intraday VWAP (Rolling 24h) as a proxy.
        2. Daily Volume Profile (Yesterday's POC, VAH, VAL).
        """
        # --- 1. Intraday VWAP (Existing) ---
        # Use time-based rolling
        window = f'{self.window_hours}h'
        pv = df["close"] * df["volume"]
        rolling_pv = pv.rolling(window=window).sum()
        rolling_vol = df["volume"].rolling(window=window).sum()

        df["vp_vwap"] = rolling_pv / rolling_vol
        df["dist_to_vwap"] = (df["close"] - df["vp_vwap"]) / df["vp_vwap"]
        df["above_vwap"] = (df["close"] > df["vp_vwap"]).astype(int)

        # --- 2. Daily Volume Profile (New) ---
        df = self._add_daily_profile(df)

        return df

    def _add_daily_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes yesterday's POC, VAH, VAL and maps them to today's candles.
        """
        if df.empty:
            return df

        # Resample to Daily to group candles
        # We need to iterate over days to compute profile for each day
        # This is slightly slow but acceptable for 1 year of data (~365 iterations)

        daily_stats = []

        # Group by date
        # Assuming index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        grouped = df.groupby(df.index.date)

        for date, group in grouped:
            if group.empty:
                continue

            # Compute Profile for this day using Numba
            # Convert to numpy arrays for Numba
            closes = group["close"].values.astype(np.float64)
            volumes = group["volume"].values.astype(np.float64)

            poc, vah, val = calculate_value_area_numba(closes, volumes, self.bins)

            daily_stats.append({"date": date, "poc": poc, "vah": vah, "val": val})

        if not daily_stats:
            return df

        # Create DataFrame from stats
        stats_df = pd.DataFrame(daily_stats)
        stats_df["date"] = pd.to_datetime(stats_df["date"])
        stats_df.set_index("date", inplace=True)

        # Shift by 1 day (Yesterday's profile applies to Today)
        stats_df = stats_df.shift(1)

        # Rename columns
        stats_df.columns = ["poc_prev_day", "vah_prev_day", "val_prev_day"]

        # Merge back to original DF
        # Optimized: Use map instead of merge to avoid copying full DataFrame
        temp_date = df.index.normalize()
        
        df["poc_prev_day"] = temp_date.map(stats_df["poc_prev_day"])
        df["vah_prev_day"] = temp_date.map(stats_df["vah_prev_day"])
        df["val_prev_day"] = temp_date.map(stats_df["val_prev_day"])

        # Derived Features
        df["dist_to_poc_prev"] = (df["close"] - df["poc_prev_day"]) / df["poc_prev_day"]
        df["dist_to_vah_prev"] = (df["close"] - df["vah_prev_day"]) / df["vah_prev_day"]
        df["dist_to_val_prev"] = (df["close"] - df["val_prev_day"]) / df["val_prev_day"]

        df["inside_value_prev"] = ((df["close"] >= df["val_prev_day"]) & (df["close"] <= df["vah_prev_day"])).astype(
            int
        )
        df["above_vah_prev"] = (df["close"] > df["vah_prev_day"]).astype(int)
        df["below_val_prev"] = (df["close"] < df["val_prev_day"]).astype(int)

        return df
