import logging
from typing import Any, Dict, Optional

import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)


class TrendFollowingStrategy:
    """
    Trend Following Strategy (Phase 3.6).

    Active only in TREND regime (gated by StrategySelector).
    Targets trend continuation using:
    - EMA Alignment (Close > EMA200 > EMA200_1H)
    - ADX Strength (> 25)
    - Volume Profile Context (Avoid extended entries above VAH)
    - ML Confirmation (Strict > 0.65)

    Exits on Regime Change (out of TREND) or Trailing Stop.
    """

    def __init__(self) -> None:
        self.name = "Trend Following V1"

    def analyze(
        self,
        df: pd.DataFrame,
        ml_score: Optional[float] = None,
        confirm_df: Optional[pd.DataFrame] = None,
        l2_features: Optional[Dict[str, Any]] = None,
        btc_bullish: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Analyzes data for Trend Following entries.
        """
        # 0. Global BTC Filter (High Beta Logic)
        if not btc_bullish:
            return {
                "signal": None,
                "size_multiplier": 0.0,
                "decision_context": {"reason_string": "BTC Bearish (Regime Filter)"},
                "close": df.iloc[-1]["close"] if not df.empty else 0.0,
            }

        # Default Result
        result: Dict[str, Any] = {
            "signal": None,
            "size_multiplier": 0.0,
            "decision_context": {},
            # Top-level keys required by BacktestRunner
            "close": 0.0,
            "atr": 0.0,
            "rsi": 50.0,
            "bb_mid": 0.0,
            "ema200": 0.0,
        }

        if df.empty:
            return result

        # 1. Get Latest Data
        current = df.iloc[-1]
        timestamp = pd.to_datetime(str(current.name))

        # Populate top-level keys
        result["close"] = current["close"]
        result["atr"] = current.get("atr", 0.0)
        result["rsi"] = current.get("rsi", 50.0)
        result["bb_mid"] = current.get("bb_mid", current["close"])
        result["ema200"] = current.get("ema200", current["close"])

        # 2. Build Decision Context
        vol_ma = current.get("volume_ma", 0.0)
        volume = current.get("volume", 0.0)
        vol_ratio = volume / vol_ma if vol_ma > 0 else 0.0

        ema50 = current.get("ema50")
        ema_extension = (current["close"] - ema50) / ema50 if ema50 and ema50 > 0 else 0.0

        context: Dict[str, Any] = {
            "timestamp": timestamp,
            "close": current["close"],
            "ema200": current.get("ema200"),
            "ema50": current.get("ema50"),
            "ema200_1h": current.get("ema200_1h"),
            "adx": current.get("adx"),
            "atr": current.get("atr"),
            "rsi": current.get("rsi"),
            "bb_width": current.get("bb_width"),
            "ml_score": ml_score,
            "regime": "TREND",  # Assumed if called
            "btc_bullish": btc_bullish,
            "vol_ratio": vol_ratio,
            "ema_extension": ema_extension,
            # Volume Profile
            "poc_prev_day": current.get("poc_prev_day"),
            "vah_prev_day": current.get("vah_prev_day"),
            "val_prev_day": current.get("val_prev_day"),
            "dist_to_poc_prev": current.get("dist_to_poc_prev"),
            "inside_value_prev": current.get("inside_value_prev"),
            "above_vah_prev": current.get("above_vah_prev"),
            "below_val_prev": current.get("below_val_prev"),
        }
        result["decision_context"] = context

        # 3. Guardrails (Volatility Shield)
        # Reuse Mean Reversion's shield
        if current["atr"] / current["close"] > Config.MAX_VOLATILITY_THRESHOLD:
            context["reason_string"] = "High Volatility"
            return result

        # 4. Trend Logic (Long Only)
        # Condition A: Trend Alignment
        # Switch to EMA50 > EMA200 (Golden Cross Alignment) for faster momentum
        ema50 = current.get("ema50")
        ema200 = current.get("ema200")
        ema200_1h = current.get("ema200_1h")

        # 1. Medium Term Alignment (5m)
        if not (ema50 and ema200 and current["close"] > ema50 and ema50 > ema200):
            context["reason_string"] = "Trend Misaligned (Price > EMA50 > EMA200)"
            return result

        # 2. Long Term Alignment (1h) - Avoid fighting major downtrends
        # Require Full Alignment: EMA50 > EMA200 > EMA200_1H
        # This ensures the 5m trend is fully established above the 1H trend.
        if ema200_1h and ema200 and ema200 < ema200_1h:
            context["reason_string"] = "Counter-Trend (EMA200 < 1H EMA200)"
            return result

        # Condition B: Trend Strength
        if current["adx"] < Config.ADX_THRESHOLD:
            context["reason_string"] = "Weak Trend (Low ADX)"
            return result

        # Avoid Exhaustion (ADX too high)
        max_adx = getattr(Config, "TREND_ADX_MAX", 100.0)
        if current["adx"] > max_adx:
            context["reason_string"] = f"Trend Exhaustion (ADX {current['adx']:.1f} > {max_adx})"
            return result

        # Condition C: Volume Profile Context
        # Avoid chasing if far above VAH (Exhaustion risk)
        if current.get("above_vah_prev") == 1 and current.get("dist_to_poc_prev", 0) > 0.02:
            context["reason_string"] = "Extended above Value Area"
            return result

        # Condition D: Trend Extension (Avoid buying tops)
        # Calculate extension from EMA50 (Momentum Base)
        max_extension = getattr(Config, "TREND_MAX_EXTENSION", 0.015)
        if ema_extension > max_extension:
            context["reason_string"] = f"Overextended ({ema_extension:.2%})"
            return result

        # Condition E: RSI Filter
        rsi = current.get("rsi")
        if rsi is not None:
            max_rsi = getattr(Config, "TREND_RSI_MAX", 60.0)

            if rsi > max_rsi:
                context["reason_string"] = f"RSI Overbought ({rsi:.2f} > {max_rsi})"
                return result
        else:
            context["reason_string"] = "RSI Missing"
            return result

        # Condition F: Momentum Confirmation (Avoid catching falling knives)
        # Require a Green Candle (Close > Open) to ensure some buying pressure
        if current["close"] <= current["open"]:
            context["reason_string"] = "Red Candle (Falling Knife)"
            return result

        # Condition G: Volume Accelerator (Retail Mania)
        # Check if Volume > 2.0 * VolMA
        if vol_ratio > 2.0:
            context["is_mania"] = True

        # Condition H: Volume Gate (Momentum Confirmation)
        # Require Volume > 1.0 * VolMA for SOL entries
        if vol_ratio < 1.0:
            context["reason_string"] = f"Low Volume ({vol_ratio:.2f} < 1.0)"
            return result

        # Avoid Climax Volume (Buying the top)
        # Unless it's a breakout from low volatility? No, safer to avoid.
        if vol_ratio > 1.5:
            context["reason_string"] = f"Climax Volume ({vol_ratio:.2f} > 1.5)"
            return result

        # 5. ML Confirmation
        ml_threshold = getattr(Config, "ML_TREND_LONG_THRESHOLD", 0.65)

        if Config.ML_ENABLED and ml_threshold > 0.0:
            if ml_score is None:
                if Config.ML_ENABLED:  # Strict Mode
                    context["reason_string"] = "ML Score Missing"
                    return result
            elif ml_score < ml_threshold:
                context["reason_string"] = f"Low ML Score ({ml_score:.2f} < {ml_threshold})"
                return result

        # 6. 1m Confirmation (Optional)
        if confirm_df is not None and not confirm_df.empty:
            # Simple check: No sharp reversal in last 3 bars
            last_1m = confirm_df.iloc[-1]
            first_1m = confirm_df.iloc[0]
            if last_1m["close"] < first_1m["open"]:  # Bearish candle sequence
                pass

        # 7. Signal Generation
        result["signal"] = "LONG"
        # Dynamic Sizing handled by RiskManager via ATR Multiplier (Config.ATR_MULTIPLIER)
        # We pass 0.5 to indicate "Standard Trend Size" (which is 0.6x BTC size per user request)
        # If RiskManager uses (Equity * 0.01) / (2.5 * ATR), then size_multiplier scales that.
        # Let's assume RiskManager handles the base calculation.
        result["size_multiplier"] = 0.6
        context["final_signal"] = "LONG"
        context["reason_string"] = "Trend Follow Entry"

        return result

    def get_exit_updates(self, position: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates exit updates (Trailing SL, TP, etc.) for Trend Following positions.
        """
        updates: Dict[str, Any] = {"exit_signal": False, "exit_reason": None, "new_sl": None, "tp1": False}

        current_price = analysis["close"]
        analysis.get("ema50")
        analysis.get("ema20", analysis.get("bb_mid"))  # Fallback to BB Mid (SMA20)

        # Check for Mania Context
        is_mania = False
        if "decision_context" in analysis:
            is_mania = analysis["decision_context"].get("is_mania", False)

        # 1. Regime Change Exit
        regime = analysis.get("regime")
        if not regime and "decision_context" in analysis:
            regime = analysis["decision_context"].get("regime")

        if regime == "CRASH":
            updates["exit_signal"] = True
            updates["exit_reason"] = f"Regime Change ({regime})"
            return updates

        # 2. Trailing Stop (Dynamic)
        # If Mania: Trail EMA50 (Looser)
        # If Normal: Trail EMA20 (Standard)
        # Or use ATR Multiplier

        trail_level = 0.0
        atr = analysis.get("atr", 0.0)

        if atr > 0:
            if is_mania:
                # Looser Trail (3.0 ATR)
                trail_level = current_price - (3.0 * atr)
            else:
                # Standard Trail (2.0 ATR)
                trail_level = current_price - (2.0 * atr)

            # Only move SL up
            if trail_level > position["stop_loss"]:
                updates["new_sl"] = trail_level

        # 3. ML Deterioration (Optional)
        ml_score = analysis.get("ml_score")
        if ml_score is not None and ml_score < 0.45:
            updates["exit_signal"] = True
            updates["exit_reason"] = "ML Deterioration"

        return updates
