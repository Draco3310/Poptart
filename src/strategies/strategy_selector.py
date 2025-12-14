import logging
from typing import Any, Dict, Optional

import pandas as pd

from src.predictors.regime_classifier import MarketRegime, RegimeClassifier
from src.strategies.btc_smart_dca import BTCSmartDCAStrategy
from src.strategies.mean_reversion_mtf import MeanReversionStrategy
from src.strategies.trend_following import TrendFollowingStrategy

logger = logging.getLogger(__name__)


class StrategySelector:
    """
    Orchestrates strategy selection based on the Market Regime.

    Routing Logic:
    - RANGE: Routes to MeanReversionStrategy (Primary).
    - TREND: Routes to TrendFollowingStrategy (Primary).
             Falls back to MeanReversionStrategy (Counter-Trend Sniper) if TF has no signal.
    - CRASH: Routes to MeanReversionStrategy (Defensive Mode - Exits Only).

    Tags trades with 'active_strategy' for analytics attribution.
    """

    def __init__(self, regime_model_path: Optional[str] = None):
        # If None, RegimeClassifier uses default or heuristic.
        # We pass it explicitly if provided.
        # NOTE: If regime_model_path is None, we pass None to force Heuristic Mode.
        # If we called RegimeClassifier() without args, it would load the default model.
        self.regime_classifier = RegimeClassifier(model_path=regime_model_path)

        self.mean_reversion = MeanReversionStrategy()
        self.trend_following = TrendFollowingStrategy()
        self.btc_dca = BTCSmartDCAStrategy()

        self.last_regime: Optional[MarketRegime] = None

    def analyze(
        self,
        df: pd.DataFrame,
        ml_score: Optional[float] = None,
        confirm_df: Optional[pd.DataFrame] = None,
        l2_features: Optional[Dict[str, float]] = None,
        enable_mean_reversion: bool = True,
        enable_trend_following: bool = True,
        enable_dca_mode: bool = False,
        dca_config: Optional[Dict[str, Any]] = None,
        current_price: float = 0.0,
        current_balance_btc: float = 0.0,
        current_equity_usdt: float = 0.0,
        regime: Optional[str] = None,  # Allow passing explicit regime
        adx_threshold: Optional[float] = None,  # Pass ADX threshold for heuristic
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        1. Detect Regime (unless DCA mode).
        2. Select Strategy (Waterfall Priority).
        3. Execute Strategy Analysis.
        """
        # 0. DCA Mode Bypass
        if enable_dca_mode and dca_config:
            return self.btc_dca.analyze(df, current_price, current_balance_btc, current_equity_usdt, dca_config)

        # 1. Detect Regime
        regime_name = "UNKNOWN"

        if regime:
            # Use provided regime
            regime_name = regime
            # Try to update internal state if possible (best effort mapping)
            try:
                self.last_regime = MarketRegime[regime]
            except Exception:
                pass
        else:
            # Calculate regime with hysteresis
            regime_enum = self.regime_classifier.predict(df, self.last_regime, adx_threshold)
            self.last_regime = regime_enum
            regime_name = regime_enum.name

        # logger.debug(f"Detected Market Regime: {regime_name}")

        # 2. Route Strategy
        # Populate basic data to prevent KeyErrors in downstream consumers (e.g. exit logic)
        curr = df.iloc[-1]
        result = {
            "signal": None,
            "size_multiplier": 0.0,
            "decision_context": {
                "regime": regime_name,
                "reason_string": f"Regime: {regime_name}",
            },
            "active_strategy": None,
            # Basic Data
            "close": curr["close"],
            "bb_mid": curr.get("bb_mid", curr["close"]),
            "atr": curr.get("atr", 0.0),
            "rsi": curr.get("rsi", 50.0),
            "ema200": curr.get("ema200", curr["close"]),
        }

        # Handle UNKNOWN regime by defaulting to RANGE (Safe Fallback)
        if regime_name == "UNKNOWN":
            # logger.warning("Regime UNKNOWN. Defaulting to RANGE logic.")
            regime_name = "RANGE"

        if regime_name == "RANGE":
            # Priority 1: Mean Reversion (if enabled)
            if enable_mean_reversion:
                result = self.mean_reversion.analyze(df, ml_score, confirm_df, l2_features, **kwargs)
                result["active_strategy"] = "MeanReversion"

            # Priority 2: Trend Following (Breakout) if MR failed or disabled
            # ONLY if we want to allow Breakout trading in Range.
            # For strict Trend Following, we should block this.
            # But if MR is disabled (e.g. SOL), we might want to catch breakouts.
            # However, data shows this loses money (-6% on SOL).
            # So we will BLOCK Trend Following in RANGE unless explicitly allowed?
            # For now, let's BLOCK it if MR is disabled, assuming "Trend Only" means "Trend Regime Only".

            if not result["signal"] and enable_trend_following and enable_mean_reversion:
                # Only allow "Breakout" logic if MR is enabled (Hybrid mode).
                # If MR is disabled (Trend Only mode), we strictly respect the Regime.
                # i.e. Don't trade Trend in Range.
                pass

            # If MR is disabled and it's RANGE, we do nothing (return empty result).

        elif regime_name == "TREND":
            # Priority 1: Trend Following (if enabled)
            if enable_trend_following:
                result = self.trend_following.analyze(df, ml_score, confirm_df, l2_features, **kwargs)
                result["active_strategy"] = "TrendFollowing"

            # Priority 2: Mean Reversion (Sniper) if TF failed or disabled
            if not result["signal"] and enable_mean_reversion:
                # We pass regime="TREND" so MR knows to apply strict sniper rules
                mr_result = self.mean_reversion.analyze(df, ml_score, confirm_df, l2_features, regime="TREND", **kwargs)
                if mr_result["signal"]:
                    result = mr_result
                    result["active_strategy"] = "MeanReversion"
                elif not enable_trend_following:
                    # If TF was disabled, return MR result
                    result = mr_result
                    result["active_strategy"] = "MeanReversion"

        elif regime_name == "CRASH":
            # Defensive Mode: No Longs
            # Let Mean Reversion handle it (it blocks longs in CRASH)
            result = self.mean_reversion.analyze(df, ml_score, confirm_df, l2_features, regime="CRASH", **kwargs)
            result["active_strategy"] = "MeanReversion"

        # Ensure decision context has regime
        if "decision_context" not in result:
            result["decision_context"] = {}
        result["decision_context"]["regime"] = regime_name

        return result

    def calculate_position_size(
        self,
        balance: float,
        entry_price: float,
        atr: float,
        multiplier: float = 1.0,
        regime: str = "CHOP",
    ) -> float:
        """
        Delegates position sizing to the active strategy or applies a global overlay.
        """
        # We could delegate, but since we don't know which strategy was active without passing it in,
        # we can use a default or pass the strategy instance.
        # For now, we'll assume Mean Reversion logic as baseline or use the regime to adjust.

        # Volatility Targeting Overlay
        # If Volatility Regime, reduce size
        if regime == "VOLATILITY":
            return 0.0

        # Use Mean Reversion's logic as default for now, or instantiate the correct one if we tracked it.
        # Ideally, analyze() returns the strategy instance or we store it in self.active_strategy

        # Simplified: Use Mean Reversion's calculator (it's robust)
        return self.mean_reversion.calculate_position_size(balance, entry_price, atr, multiplier)
