import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import importlib.metadata
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
        # Add Trend Context (Default 1h)
        df = self._add_mtf_features(df, period=Config.TIMEFRAME_CONTEXT)

        # --- 1. Momentum Indicators ---

        # RSI 14
        df["rsi"] = ta.rsi(df["close"], length=14)

        # ADX (14)
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx is not None:
            # ADX usually returns ADX_14, DMP_14, DMN_14
            # Simplify extraction
            adx_col = next((c for c in adx.columns if c.startswith("ADX_")), None)
            if adx_col:
                df["adx"] = adx[adx_col]

        # MACD (12, 26, 9)
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            # Dynamically find columns
            macd_col = next((c for c in macd.columns if c.startswith("MACD_")), None)
            hist_col = next((c for c in macd.columns if c.startswith("MACDh_")), None)
            signal_col = next((c for c in macd.columns if c.startswith("MACDs_")), None)
            
            if macd_col: df["macd"] = macd[macd_col]
            if hist_col: df["macd_hist"] = macd[hist_col]
            if signal_col: df["macd_signal"] = macd[signal_col]

        # --- 2. Volatility Indicators ---

        # Bollinger Bands (20, 2.0)
        bbands = ta.bbands(df["close"], length=20, std=2.0)  # type: ignore
        if bbands is not None:
            lower_col = next((c for c in bbands.columns if c.startswith("BBL")), None)
            mid_col = next((c for c in bbands.columns if c.startswith("BBM")), None)
            upper_col = next((c for c in bbands.columns if c.startswith("BBU")), None)
            
            if lower_col: df["bb_lower"] = bbands[lower_col]
            if mid_col: df["bb_mid"] = bbands[mid_col]
            if upper_col: df["bb_upper"] = bbands[upper_col]

            # Recalculate width if possible
            if "bb_upper" in df.columns and "bb_lower" in df.columns and "bb_mid" in df.columns:
                df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

        # ATR (14)
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        # Keltner Channels (20, 2.0) - Volatility-adjusted bands
        kc = ta.kc(df["high"], df["low"], df["close"], length=20, scalar=2.0)
        if kc is not None:
            lower_col = next((c for c in kc.columns if c.startswith("KCL")), None)
            mid_col = next((c for c in kc.columns if c.startswith("KCM")), None)
            upper_col = next((c for c in kc.columns if c.startswith("KCU")), None)
            
            if lower_col: df["kc_lower"] = kc[lower_col]
            if mid_col: df["kc_mid"] = kc[mid_col]
            if upper_col: df["kc_upper"] = kc[upper_col]

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
        ema_slow = 200
        ema_fast = 50
        
        if context and "pair_config" in context:
            pair_config = context["pair_config"]
            ema_slow = getattr(pair_config, "ema_period_slow", 200)
            ema_fast = getattr(pair_config, "ema_period_fast", 50)

        df["ema200"] = ta.ema(df["close"], length=ema_slow)

        # EMA (50) - Fast Trend
        df["ema50"] = ta.ema(df["close"], length=ema_fast)

        # Volume Moving Average (20)
        df["volume_ma"] = ta.sma(df["volume"], length=20)

        # Volume Relative to MA
        df["volume_rel"] = df["volume"] / df["volume_ma"].replace(0, np.nan)

        # On-Chain Activity Proxy (24 hours rolling volume average)
        # Use time-based rolling to support any timeframe (1m, 5m, etc.)
        df["onchain_activity"] = df["volume"].rolling(window='24h', min_periods=1).mean()

        # --- 3.5 On-Chain Features (Placeholder) ---
        df = self.add_onchain_features(df, context)

        return df

    def add_onchain_features(self, df: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Placeholder for fetching and merging on-chain data (e.g., from XRPSCAN).
        Currently uses volume proxies.
        """
        # Check if on-chain data is provided in context (pre-fetched)
        if context and "onchain_data" in context:
            # Merge logic would go here
            pass
            
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
            # 1. Trend: EMA 200 (Use default 200 as we don't have context here easily without refactoring signature)
            # Ideally we should pass context to _add_mtf_features too, but for now hardcoding 200 is safer than breaking signature
            # or we can use Config.EMA_PERIOD_SLOW if it existed, but it doesn't.
            # Let's assume 200 for MTF context as it's standard.
            resampled[f"ema200_{period}"] = ta.ema(resampled["close"], length=200)

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
