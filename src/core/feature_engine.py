import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from src.core.features.base import FeatureBlock
from src.core.features.beta import BetaFeatureBlock
from src.core.features.legacy import LegacyFeatureBlock
from src.core.features.microstructure import MicrostructureFeatureBlock
from src.core.features.statistical import StatisticalFeatureBlock
from src.core.features.transformed import TransformedFeatureBlock
from src.core.features.volume_profile import VolumeProfileFeatureBlock

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Centralized Feature Engineering Orchestrator.
    Manages a pipeline of FeatureBlocks to enrich raw OHLCV data.
    """

    def __init__(self) -> None:
        self.blocks: List[FeatureBlock] = []
        self._register_default_blocks()

    def _register_default_blocks(self) -> None:
        """
        Registers the default set of feature blocks.
        """
        # 1. Legacy Indicators (MTF, RSI, MACD, BB, etc.)
        self.blocks.append(LegacyFeatureBlock())

        # 2. Transformed Features (Z-Scores, Dist-to-MA)
        self.blocks.append(TransformedFeatureBlock())

        # 3. Statistical Features (Hurst, Autocorrelation)
        self.blocks.append(StatisticalFeatureBlock())

        # 4. Microstructure Features (VPIN, Volume Imbalance)
        self.blocks.append(MicrostructureFeatureBlock())

        # 5. Volume Profile Features (VWAP, Dist to VWAP)
        self.blocks.append(VolumeProfileFeatureBlock())

        # 6. Beta Features (High-Beta, RS, Correlation)
        self.blocks.append(BetaFeatureBlock())

    def compute_features(self, df: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Computes technical indicators and prepares features for the model.
        Runs the DataFrame through all registered FeatureBlocks.

        Args:
            df: Raw OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                and a DatetimeIndex (or convertible index).
            context: Optional dictionary containing external data (e.g., 'btc_df').

        Returns:
            Enriched DataFrame with indicators, no NaNs.
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to compute_features.")
            return df

        # Ensure we are working on a copy
        df = df.copy()

        # Ensure DatetimeIndex for Resampling
        if "timestamp" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

        # Run Pipeline
        for block in self.blocks:
            try:
                df = block.apply(df, context=context)
                
                # Only scan for NaNs in DEBUG mode to save performance (O(N*M))
                if logger.isEnabledFor(logging.DEBUG):
                    nan_count = df.isna().sum().sum()
                    logger.debug(f"DEBUG: After {block.__class__.__name__}, Rows: {len(df)}, NaNs: {nan_count}")
                    if nan_count > 0:
                        # Log which columns have NaNs
                        nan_cols = df.columns[df.isna().any()].tolist()
                        logger.debug(f"DEBUG: NaN Columns: {nan_cols}")
            except Exception as e:
                logger.error(f"Error in feature block {block.__class__.__name__}: {e}")

        # --- Data Cleaning & Imputation (Final Step) ---
        # Forward fill small gaps
        df = df.ffill()

        # Drop remaining NaNs (Warmup Period)
        # Note: We do NOT bfill() because that would propagate future data backwards (lookahead bias)
        # for indicators like EMA200 that have long warmup periods.
        original_len = len(df)
        df = df.dropna()
        dropped_len = original_len - len(df)

        if dropped_len > 0:
            logger.info(f"Dropped {dropped_len} rows due to warmup/NaNs.")

        return df

    def compute_l2_features(self, order_book: Dict[str, Any]) -> Dict[str, float]:
        """
        Computes Level 2 features from the Order Book.
        Features:
        - obi (Order Book Imbalance): (Bid Vol - Ask Vol) / (Bid Vol + Ask Vol)
        - spread: (Best Ask - Best Bid) / Best Bid
        - market_depth_ratio: Total Bid Vol / Total Ask Vol
        """
        try:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])

            if not bids or not asks:
                return {"obi": 0.0, "spread": 0.0, "market_depth_ratio": 1.0}

            # Calculate Total Volume (Depth)
            bid_vol = sum([b[1] for b in bids])
            ask_vol = sum([a[1] for a in asks])

            # 1. Order Book Imbalance (OBI)
            # Range: -1.0 (Full Sell Pressure) to 1.0 (Full Buy Pressure)
            total_vol = bid_vol + ask_vol
            obi = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0

            # 2. Spread
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0.0

            # 3. Market Depth Ratio
            # > 1.0 means more buy support
            market_depth_ratio = bid_vol / ask_vol if ask_vol > 0 else 1.0

            return {
                "obi": float(obi),
                "spread": float(spread),
                "market_depth_ratio": float(market_depth_ratio),
            }

        except Exception as e:
            logger.error(f"Error computing L2 features: {e}")
            return {"obi": 0.0, "spread": 0.0, "market_depth_ratio": 1.0}
