from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.core.features.base import FeatureBlock


class MicrostructureFeatureBlock(FeatureBlock):
    """
    Adds microstructure proxies to estimate order flow toxicity and liquidity.
    Includes VPIN Proxy and Volume Imbalance.
    """

    def apply(self, df: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()

        # 1. Volume Imbalance Proxy
        # (Close - Open) / (High - Low) * Volume
        # Range: -Volume (Full Bear) to +Volume (Full Bull)
        epsilon = 1e-9
        high_low = df["high"] - df["low"]
        range_hl = high_low.replace(0, np.nan)
        df["volume_imbalance"] = ((df["close"] - df["open"]) / range_hl) * df["volume"]

        # 2. VPIN Proxy (Volume-Synchronized Probability of Informed Trading)
        # Real VPIN requires trade-by-trade data.
        # Proxy: Ratio of "Toxic" Volume to Total Volume.
        # Toxic Volume ~ Volume * |Price Change| / Price
        # We calculate a rolling average of this toxicity.

        price_change_pct = df["close"].pct_change(fill_method=None).abs()
        toxic_flow = df["volume"] * price_change_pct

        # Use time-based window for consistency across timeframes
        window_time = '1h'
        total_volume = df["volume"].rolling(window=window_time).sum()
        total_toxic = toxic_flow.rolling(window=window_time).sum()

        df["vpin_proxy"] = total_toxic / total_volume.replace(0, np.nan)

        # 3. Candle Shape Features
        # Measures of wick/body proportions to detect rejection/exhaustion.
        # Epsilon to avoid division by zero
        candle_range = high_low + epsilon
        body_size = (df["close"] - df["open"]).abs()
        upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
        lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

        df["ms_body_to_range"] = body_size / candle_range
        df["ms_upper_wick_to_range"] = upper_wick / candle_range
        df["ms_lower_wick_to_range"] = lower_wick / candle_range
        df["ms_upper_vs_lower_wick"] = (upper_wick - lower_wick) / candle_range

        # 4. Range & Volatility Features
        # Detect range expansion/compression relative to history.
        window_range = 20 # Keep integer window for Z-Score to ensure statistical significance (N samples)
        true_range = high_low  # Simple TR for now
        r_tr = true_range.rolling(window=window_range)
        rolling_mean_tr = r_tr.mean()
        rolling_std_tr = r_tr.std()

        df[f"ms_range_zscore_{window_range}"] = (true_range - rolling_mean_tr) / (rolling_std_tr + epsilon)

        if "atr" in df.columns:
            df["ms_range_to_atr"] = true_range / (df["atr"] + epsilon)

        # Parkinson Volatility (High/Low based)
        # Captures "Wick" volatility better than Close-to-Close
        # Formula: sqrt(1 / (4 * ln(2)) * mean(ln(H/L)^2))
        # We calculate rolling Parkinson Volatility
        const = 1.0 / (4.0 * np.log(2.0))
        log_hl = np.log(df["high"] / df["low"].replace(0, np.nan))
        log_hl_sq = pd.Series(log_hl**2, index=df.index)

        # Rolling mean of squared log range
        rolling_log_hl_sq = log_hl_sq.rolling(window=window_range).mean()
        df["parkinson_vol"] = np.sqrt(const * rolling_log_hl_sq)

        # 5. Volume Features
        # Detect volume spikes and efficiency.
        window_vol = 20
        r_vol = df["volume"].rolling(window=window_vol)
        rolling_mean_vol = r_vol.mean()
        rolling_std_vol = r_vol.std()

        df[f"vol_zscore_{window_vol}"] = (df["volume"] - rolling_mean_vol) / (rolling_std_vol + epsilon)

        # Binary flag for volume spikes (> 2.0 sigma)
        df[f"vol_spike_flag_{window_vol}"] = (df[f"vol_zscore_{window_vol}"] > 2.0).astype(float)

        # Volume per unit of range (Liquidity/Absorption proxy)
        df["vol_per_range"] = df["volume"] / candle_range

        # 6. VWAP & Value Area Proxies (Rolling 24h)
        # Efficient calculation of Rolling VWAP
        # Use time-based rolling to support any timeframe
        window_vwap = '24h'

        # Typical Price
        tp = (df["high"] + df["low"] + df["close"]) / 3
        tp_vol = tp * df["volume"]

        rolling_tp_vol = tp_vol.rolling(window=window_vwap).sum()
        rolling_vol = df["volume"].rolling(window=window_vwap).sum()

        df["vwap_24h"] = rolling_tp_vol / rolling_vol.replace(0, np.nan)

        # Distance to VWAP (Normalized by ATR if available, else Close)
        if "atr" in df.columns:
            df["dist_to_vwap_atr"] = (df["close"] - df["vwap_24h"]) / (df["atr"] + epsilon)
        else:
            df["dist_to_vwap_pct"] = (df["close"] - df["vwap_24h"]) / df["vwap_24h"]

        # VWAP Bands (Standard Deviation of Price Volume Distribution)
        # StdDev = sqrt( VWAP(Price^2) - VWAP(Price)^2 )
        tp_sq_vol = (tp ** 2) * df["volume"]
        rolling_tp_sq_vol = tp_sq_vol.rolling(window=window_vwap).sum()
        
        vwap_sq = rolling_tp_sq_vol / rolling_vol.replace(0, np.nan)
        # Variance = E[X^2] - (E[X])^2
        # Ensure non-negative (floating point errors)
        variance = (vwap_sq - (df["vwap_24h"] ** 2)).clip(lower=0)
        vwap_std = np.sqrt(variance)

        df["vwap_upper"] = df["vwap_24h"] + (2.0 * vwap_std)
        df["vwap_lower"] = df["vwap_24h"] - (2.0 * vwap_std)

        # Feature: Position within VWAP Bands (0.0 = Lower, 0.5 = VWAP, 1.0 = Upper)
        vwap_range = df["vwap_upper"] - df["vwap_lower"]
        df["vwap_position"] = (df["close"] - df["vwap_lower"]) / (vwap_range + epsilon)

        return df
