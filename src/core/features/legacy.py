import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import importlib.metadata  # Fix for pandas_ta AttributeError
import pandas_ta as ta

from src.config import Config
from src.core.features.base import FeatureBlock

logger = logging.getLogger(__name__)


class LegacyFeatureBlock(FeatureBlock):
    """
    Implements the original set of indicators from Poptart V2.
    Includes MTF, Momentum, Volatility, and Volume indicators.
    """

    def apply(self, df: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()

        # --- 0. Multi-Timeframe Features (MTF) ---
        # Add 1-Hour Trend Context
        df = self._add_mtf_features(df, period="1h")

        # --- 1. Momentum Indicators ---

        # RSI 14
        df["rsi"] = ta.rsi(df["close"], length=14)

        # ADX (14)
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx is not None:
            # ADX usually returns ADX_14, DMP_14, DMN_14
            for col in adx.columns:
                if col.startswith("ADX_"):
                    df["adx"] = adx[col]

        # MACD (12, 26, 9)
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            # Dynamically find columns
            for col in macd.columns:
                if col.startswith("MACD_"):
                    df["macd"] = macd[col]
                elif col.startswith("MACDh_"):
                    df["macd_hist"] = macd[col]
                elif col.startswith("MACDs_"):
                    df["macd_signal"] = macd[col]

        # --- 2. Volatility Indicators ---

        # Bollinger Bands (20, 2.0)
        bbands = ta.bbands(df["close"], length=20, std=2.0)  # type: ignore
        if bbands is not None:
            for col in bbands.columns:
                if col.startswith("BBL"):
                    df["bb_lower"] = bbands[col]
                elif col.startswith("BBM"):
                    df["bb_mid"] = bbands[col]
                elif col.startswith("BBU"):
                    df["bb_upper"] = bbands[col]

            # Recalculate width if possible
            if "bb_upper" in df.columns and "bb_lower" in df.columns and "bb_mid" in df.columns:
                df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

        # ATR (14)
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        # Keltner Channels (20, 2.0) - Volatility-adjusted bands
        kc = ta.kc(df["high"], df["low"], df["close"], length=20, scalar=2.0)
        if kc is not None:
            for col in kc.columns:
                if col.startswith("KCL"):
                    df["kc_lower"] = kc[col]
                elif col.startswith("KCM"):
                    df["kc_mid"] = kc[col]
                elif col.startswith("KCU"):
                    df["kc_upper"] = kc[col]

        # Volatility Scaled Returns (Short Vol / Long Vol)
        # Using 10 and 30 periods as seen in legacy code
        df["returns"] = df["close"].pct_change(fill_method=None)
        short_vol = df["returns"].rolling(window=10).std()
        long_vol = df["returns"].rolling(window=30).std()

        # Handle division by zero if flat price
        df["volatility_ratio"] = short_vol / long_vol.replace(0, np.nan)

        # --- 3. Volume Indicators ---

        # EMA (200) - Trend Filter
        # Note: We keep the column name 'ema200' for ML compatibility, but use the configured period
        df["ema200"] = ta.ema(df["close"], length=Config.EMA_PERIOD_SLOW)

        # EMA (50) - Fast Trend
        df["ema50"] = ta.ema(df["close"], length=Config.EMA_PERIOD_FAST)

        # Volume Moving Average (20)
        df["volume_ma"] = ta.sma(df["volume"], length=20)

        # Volume Relative to MA
        df["volume_rel"] = df["volume"] / df["volume_ma"].replace(0, np.nan)

        # On-Chain Activity Proxy (1440 min = 24 hours rolling volume average)
        df["onchain_activity"] = df["volume"].rolling(window=1440, min_periods=1).mean()

        # --- 3.5 On-Chain Features (Placeholder) ---
        df = self.add_onchain_features(df)

        return df

    def add_onchain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Placeholder for fetching and merging on-chain data (e.g., from XRPSCAN).
        Currently uses volume proxies.
        """
        # Future: Fetch real on-chain data here
        # For now, we ensure 'tx_volume' exists as a feature for ML models
        if "tx_volume" not in df.columns:
            # Proxy: Use Volume * Close as transaction volume estimate
            df["tx_volume"] = df["volume"] * df["close"]

        return df

    def _add_mtf_features(self, df: pd.DataFrame, period: str = "1h") -> pd.DataFrame:
        """
        Resamples data to a higher timeframe, calculates indicators,
        and merges them back to the original DataFrame.
        """
        try:
            # Resample
            agg_rules = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            resampled = df.resample(period).agg(agg_rules)  # type: ignore

            # Calculate Indicators on Higher Timeframe
            # 1. Trend: EMA 200
            resampled[f"ema200_{period}"] = ta.ema(resampled["close"], length=Config.EMA_PERIOD_SLOW)

            # 2. Momentum: RSI 14 (Optional, for confluence)
            resampled[f"rsi_{period}"] = ta.rsi(resampled["close"], length=14)

            # Merge back to original DF
            # We use forward fill to propagate the last known 1h value to all 1m candles in that hour
            # Reindex to match original df index
            resampled = resampled.reindex(df.index, method="ffill")

            # Add columns to df
            df[f"ema200_{period}"] = resampled[f"ema200_{period}"]
            df[f"rsi_{period}"] = resampled[f"rsi_{period}"]

        except Exception as e:
            logger.error(f"Error computing MTF features for {period}: {e}")

        return df
