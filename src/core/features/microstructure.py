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
        range_hl = (df["high"] - df["low"]).replace(0, np.nan)
        df["volume_imbalance"] = ((df["close"] - df["open"]) / range_hl) * df["volume"]

        # 2. VPIN Proxy (Volume-Synchronized Probability of Informed Trading)
        # Real VPIN requires trade-by-trade data.
        # Proxy: Ratio of "Toxic" Volume to Total Volume.
        # Toxic Volume ~ Volume * |Price Change| / Price
        # We calculate a rolling average of this toxicity.

        price_change_pct = df["close"].pct_change(fill_method=None).abs()
        toxic_flow = df["volume"] * price_change_pct

        window = 20
        total_volume = df["volume"].rolling(window=window).sum()
        total_toxic = toxic_flow.rolling(window=window).sum()

        df["vpin_proxy"] = total_toxic / total_volume.replace(0, np.nan)

        # 3. Candle Shape Features
        # Measures of wick/body proportions to detect rejection/exhaustion.
        # Epsilon to avoid division by zero
        epsilon = 1e-9
        candle_range = (df["high"] - df["low"]) + epsilon
        body_size = (df["close"] - df["open"]).abs()
        upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
        lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

        df["ms_body_to_range"] = body_size / candle_range
        df["ms_upper_wick_to_range"] = upper_wick / candle_range
        df["ms_lower_wick_to_range"] = lower_wick / candle_range
        df["ms_upper_vs_lower_wick"] = (upper_wick - lower_wick) / candle_range

        # 4. Range & Volatility Features
        # Detect range expansion/compression relative to history.
        window_range = 20
        true_range = df["high"] - df["low"]  # Simple TR for now
        rolling_mean_tr = true_range.rolling(window=window_range).mean()
        rolling_std_tr = true_range.rolling(window=window_range).std()

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
        rolling_mean_vol = df["volume"].rolling(window=window_vol).mean()
        rolling_std_vol = df["volume"].rolling(window=window_vol).std()

        df[f"vol_zscore_{window_vol}"] = (df["volume"] - rolling_mean_vol) / (rolling_std_vol + epsilon)

        # Binary flag for volume spikes (> 2.0 sigma)
        df[f"vol_spike_flag_{window_vol}"] = (df[f"vol_zscore_{window_vol}"] > 2.0).astype(float)

        # Volume per unit of range (Liquidity/Absorption proxy)
        df["vol_per_range"] = df["volume"] / candle_range

        # 6. VWAP & Value Area Proxies (Rolling 24h = 288 bars @ 5m)
        # Efficient calculation of Rolling VWAP
        window_vwap = 288

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

        # VWAP Bands (Proxy for Value Area)
        # We need rolling standard deviation of price relative to VWAP?
        # Or just standard deviation of Close?
        # Standard VWAP bands use std dev of the *price* distribution.
        # Approximation: Rolling Std Dev of Close * 2
        rolling_std_close = df["close"].rolling(window=window_vwap).std()
        df["vwap_upper"] = df["vwap_24h"] + (2.0 * rolling_std_close)
        df["vwap_lower"] = df["vwap_24h"] - (2.0 * rolling_std_close)

        # Feature: Position within VWAP Bands (0.0 = Lower, 0.5 = VWAP, 1.0 = Upper)
        vwap_range = df["vwap_upper"] - df["vwap_lower"]
        df["vwap_position"] = (df["close"] - df["vwap_lower"]) / (vwap_range + epsilon)

        return df
