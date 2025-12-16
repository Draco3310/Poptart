import logging
import os
from enum import Enum
from typing import Any, Optional

import joblib  # type: ignore
import pandas as pd

from src.config import Config
from src.predictors.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    RANGE = 0
    TREND = 1
    CRASH = 2
    UNKNOWN = 3


class RegimeClassifier:
    """
    Classifies the current market regime using a trained Random Forest model.

    Labels:
    - 0: RANGE (Low Volatility, Weak Trend)
    - 1: TREND (Strong ADX, Directional)
    - 2: CRASH (Extreme Volatility, Panic Selling)

    Features: ADX, ATR, Microstructure (z-scores), Volume Profile context.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model: Any = None
        self.features: list[str] = []

        if model_path:
            # Check Registry First
            cached = ModelRegistry.get(model_path)
            if cached:
                self.model = cached["model"]
                self.features = cached["features"]
                return

            if os.path.exists(model_path):
                try:
                    artifact = joblib.load(model_path)
                    if isinstance(artifact, dict) and "model" in artifact and "features" in artifact:
                        self.model = artifact["model"]
                        self.features = artifact["features"]
                        # Force single-threaded inference to avoid nested parallelism warnings
                        if hasattr(self.model, "n_jobs"):
                            self.model.n_jobs = 1
                        logger.info(f"Loaded Regime Classifier and {len(self.features)} features from {model_path}")

                        # Register it
                        ModelRegistry.register(model_path, {"model": self.model, "features": self.features})
                    else:
                        # Legacy fallback (if model saved directly)
                        self.model = artifact
                        if hasattr(self.model, "n_jobs"):
                            self.model.n_jobs = 1
                        logger.warning("Loaded legacy Regime Classifier (no feature list). Using default features.")
                        self.features = [
                            "adx",
                            "atr",
                            "rsi",
                            "ema200",
                            "ema200_1h",
                            "bb_width",
                            "vol_zscore_20",
                            "ms_range_zscore_20",
                            "ms_range_to_atr",
                            "vwap_position",
                            "vpin_proxy",
                            "volume_imbalance",
                            "dist_to_vwap_atr",
                        ]
                        # Register it
                        ModelRegistry.register(model_path, {"model": self.model, "features": self.features})
                except Exception as e:
                    logger.error(f"Failed to load Regime Classifier: {e}")
            else:
                logger.warning(f"Regime Classifier model not found at {model_path}. Using Heuristic Mode.")
        else:
            logger.info("No model path provided. Using Heuristic Mode.")

    def predict(
        self, df: pd.DataFrame, previous_regime: Optional[MarketRegime] = None, adx_threshold: Optional[float] = None
    ) -> MarketRegime:
        """
        Determines the regime for the latest candle.
        """
        if df.empty:
            return MarketRegime.UNKNOWN

        if self.model is None:
            return self._predict_heuristic(df, previous_regime, adx_threshold)

        # Prepare single row DataFrame
        curr = df.iloc[[-1]].copy()

        # Ensure derived features exist (must match label_regimes.py)
        # Modify in-place to avoid redundant copies
        if "volatility" not in curr.columns and "atr" in curr.columns and "close" in curr.columns:
            curr["volatility"] = curr["atr"] / curr["close"]

        if "ema_bias" not in curr.columns and "ema200" in curr.columns and "close" in curr.columns:
            curr["ema_bias"] = (curr["close"] - curr["ema200"]) / curr["ema200"]

        # Check for missing features
        missing = [f for f in self.features if f not in curr.columns]
        if missing:
            # logger.debug(f"Missing features for Regime Classification: {missing}")
            return MarketRegime.UNKNOWN

        try:
            X = curr[self.features]
            pred = self.model.predict(X)[0]

            # DEBUG: Log prediction details occasionally
            # if pred != 0:
            #    logger.info(f"Regime Prediction: {pred} (Features: {X.values})")

            # Map prediction (0, 1, 2) to Enum
            if pred == 0:
                return MarketRegime.RANGE
            elif pred == 1:
                return MarketRegime.TREND
            elif pred == 2:
                return MarketRegime.CRASH
            else:
                return MarketRegime.UNKNOWN
        except Exception as e:
            logger.error(f"Regime prediction failed: {e}")
            return MarketRegime.UNKNOWN

    def _predict_heuristic(
        self, df: pd.DataFrame, previous_regime: Optional[MarketRegime] = None, adx_threshold: Optional[float] = None
    ) -> MarketRegime:
        """
        Simple heuristic fallback when no ML model is available.
        Rules with Hysteresis:
        - Enter TREND if ADX > Threshold + Hysteresis
        - Exit TREND if ADX < Threshold - Hysteresis
        """
        try:
            curr = df.iloc[-1]
            adx = curr.get("adx", 0)

            # Use provided threshold or fallback to global config
            threshold = adx_threshold if adx_threshold is not None else getattr(Config, "ADX_THRESHOLD", 25)
            hysteresis = getattr(Config, "REGIME_HYSTERESIS", 5.0)

            logger.debug(f"Heuristic Regime Check: ADX={adx:.2f}, Threshold={threshold}, Prev={previous_regime}")

            if previous_regime == MarketRegime.TREND:
                # Harder to exit TREND
                if adx < (threshold - hysteresis):
                    return MarketRegime.RANGE
                else:
                    return MarketRegime.TREND
            else:
                # Harder to enter TREND (or initial state)
                if adx > (threshold + hysteresis):
                    return MarketRegime.TREND
                else:
                    return MarketRegime.RANGE

        except Exception as e:
            logger.error(f"Heuristic prediction failed: {e}")
            return MarketRegime.UNKNOWN
