import logging
from typing import Any, Dict, Optional

import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)


class MeanReversionStrategy:
    """
    Mean Reversion Strategy (Multi-Timeframe).

    Primary strategy for RANGE regimes.
    Can also act as a 'Counter-Trend Sniper' in TREND regimes if ML confidence is high.

    Logic:
    - Entry: Price touches BB/KC bands + RSI extremes + ML confirmation.
    - Filters: Volatility shield, Anti-Knife (1m), L2 OBI, Volume Profile context.
    - Exit: Reversion to mean (BB Mid/EMA) or Stop Loss.
    """

    def analyze(
        self,
        df: pd.DataFrame,
        ml_score: Optional[float] = None,
        confirm_df: Optional[pd.DataFrame] = None,
        l2_features: Optional[Dict[str, float]] = None,
        regime: str = "RANGE",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Calculates indicators and returns the latest signal analysis with ML filtering.

        Args:
            df: Enriched DataFrame from FeatureEngine (Primary Timeframe).
            ml_score: Ensemble ML Prediction Score (0.0 to 1.0) or None if neutral/unavailable.
            confirm_df: Enriched DataFrame from FeatureEngine (Confirmation Timeframe - 1m).
            l2_features: Dict containing 'obi', 'spread', 'market_depth_ratio'.

        Returns:
            Dict containing signal, metadata, and position sizing multiplier.
        """
        if df.empty or len(df) < 2:
            return {"signal": None}

        # DEBUG: Log columns and first few rows to diagnose missing data
        if len(df) > 200 and len(df) < 205:
            logger.info(f"DEBUG DF COLUMNS: {list(df.columns)}")
            logger.info(f"DEBUG DF HEAD: {df.head(3)}")
            logger.info(f"DEBUG DF TAIL: {df.tail(3)}")

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # DEBUG: Log current row content
        if len(df) > 200 and len(df) < 205:
            logger.info(f"DEBUG CURR: {curr}")
            logger.info(f"DEBUG CURR OPEN: {curr.get('open', 'MISSING')}")
            logger.info(f"DEBUG CURR HIGH: {curr.get('high', 'MISSING')}")
            logger.info(f"DEBUG CURR LOW: {curr.get('low', 'MISSING')}")
            logger.info(f"DEBUG CURR CLOSE: {curr.get('close', 'MISSING')}")

        # DEBUG: Check Config.ML_ENABLED
        if len(df) > 200 and len(df) < 205:
            logger.info(f"DEBUG: Config.ML_ENABLED inside Strategy: {Config.ML_ENABLED}")
            logger.info(f"DEBUG: ml_score passed to Strategy: {ml_score}")

        # Basic data required for exit logic even if not enough for entry logic
        base_result = {
            "signal": None,
            "size_multiplier": 0.0,
            "close": curr["close"],
            "atr": curr.get("atr", 0.0),
            "rsi": curr.get("rsi", 50.0),
            "bb_mid": curr.get("bb_mid", curr["close"]),
            "ema200": curr.get("ema200", curr["close"]),
            "ml_score": ml_score,
            "kc_lower": curr.get("kc_lower", curr["close"]),
            "kc_upper": curr.get("kc_upper", curr["close"]),
            "l2_features": l2_features,
        }

        if len(df) < 200:
            return base_result

        # Capture L2 features for use throughout
        obi: Optional[float] = None
        spread: Optional[float] = None
        market_depth_ratio: Optional[float] = None
        if l2_features:
            obi = l2_features.get("obi")
            spread = l2_features.get("spread")
            market_depth_ratio = l2_features.get("market_depth_ratio")

        # --- 1. Market Regime Filter (Classifier & Volatility) ---
        # Use Regime Classifier instead of raw ADX
        adx = curr.get("adx", 0)

        # Allow trading in RANGE, or TREND if ML confidence is high
        is_valid_regime = regime == "RANGE"

        # Volatility Filter (Safety Net)
        atr = curr.get("atr", 0.0)
        close = curr["close"]
        volatility = atr / close if close > 0 else 0.0
        max_vol = getattr(Config, "MAX_VOLATILITY_THRESHOLD", 1.0)
        is_low_vol = volatility < max_vol

        # --- 2. MTF Trend Filter (1-Hour EMA 200) ---
        ema200_1h = curr.get("ema200_1h", curr.get("ema200"))
        is_uptrend_1h = curr["close"] > ema200_1h if ema200_1h else True
        is_downtrend_1h = curr["close"] < ema200_1h if ema200_1h else True

        # --- 3. On-Chain Activity Filter ---
        # Boost position size if current volume activity exceeds average by 50%
        onchain_activity = curr.get("onchain_activity", curr.get("volume_ma", 0))
        volume_ma = curr.get("volume_ma", 1)
        onchain_multiplier = 1.2 if onchain_activity > volume_ma * 1.5 else 1.0

        # --- 4. Entry Logic: Dual-Band Confirmation ---
        signal = None
        size_multiplier = 0.0

        # Decision Context Variables (for logging)
        touched_band = False
        is_green = False
        rsi_hook = False
        confirmed_1m = False
        obi_filter = False
        ml_filter = False
        pre_ml_signal: Optional[str] = None
        reason_components = []

        # DEBUG: Print Close vs BB_Mid
        bb_mid_val = curr.get("bb_mid", curr["close"])
        if curr["close"] >= bb_mid_val:
            logger.debug(f"DEBUG: Close ({curr['close']:.5f}) >= BB_Mid ({bb_mid_val:.5f}). Should NOT be LONG.")
        else:
            logger.debug(f"DEBUG: Close ({curr['close']:.5f}) < BB_Mid ({bb_mid_val:.5f}). Potential LONG.")

        if not is_valid_regime:
            # Exception: Allow TREND if ML Score is decent (Counter-Trend Sniper)
            # We relaxed this from 0.8 to 0.6 to allow more pullback trades in trends
            trend_ml_threshold = 0.6
            if regime == "TREND" and ml_score is not None and ml_score > trend_ml_threshold:
                logger.info(
                    f"Counter-Trend Sniper: Allowing trade in TREND regime due to ML score "
                    f"({ml_score:.2f} > {trend_ml_threshold})"
                )
                reason_components.append(f"Regime Override: TREND allowed by ML > {trend_ml_threshold}")
            else:
                logger.debug(f"Market regime {regime} not suitable for Mean Reversion")
                reason_components.append(f"Filtered: regime {regime}")
                # We continue to calculate other metrics for the decision log, but signal remains None

        elif not is_low_vol:
            logger.debug(f"Market volatility too high ({volatility:.4f}), skipping mean reversion signal")
            reason_components.append(f"Filtered: high volatility ({volatility:.4f} >= {max_vol})")

        else:
            # LONG: Uptrend + Touch Lower Bands (BB OR KC) + RSI Oversold
            # DEBUG: Log RSI check values
            if curr["rsi"] < Config.RSI_OVERSOLD + 5:  # Log near misses too
                logger.debug(f"DEBUG CHECK: RSI={curr['rsi']:.4f}, Threshold={Config.RSI_OVERSOLD}")

            if curr["rsi"] < Config.RSI_OVERSOLD:
                if not is_uptrend_1h:
                    logger.debug(
                        f"LONG Ignored: Not in 1H Uptrend (Close={curr['close']:.4f} < EMA200_1H={ema200_1h:.4f})"
                    )
                    reason_components.append("LONG filtered: not in 1H uptrend")
                else:
                    # Check if price touches either Bollinger Lower OR Keltner Lower (Current OR Previous)
                    band_buffer = getattr(Config, "BAND_ATR_BUFFER", 0.0) * curr.get("atr", 0.0)
                    touched_band = (
                        (curr["low"] <= curr["bb_lower"] - band_buffer)
                        or (curr["low"] <= curr.get("kc_lower", curr["bb_lower"]) - band_buffer)
                        or (prev["low"] <= prev["bb_lower"] - band_buffer)
                        or (prev["low"] <= prev.get("kc_lower", prev["bb_lower"]) - band_buffer)
                    )

                    if not touched_band:
                        logger.debug(
                            f"LONG Ignored: No Band Touch (Low={curr['low']:.4f}, BB_Low={curr['bb_lower']:.4f}, "
                            f"buffer={band_buffer:.5f})"
                        )
                        reason_components.append("LONG filtered: no band touch")
                    else:
                        # --- Anti-Knife Confirmation ---
                        is_green = curr["close"] > curr["open"]
                        rsi_hook = curr["rsi"] > prev["rsi"]
                        higher_close = curr["close"] > prev["close"]

                        if is_green and rsi_hook and higher_close:
                            logger.debug(
                                f"LONG Setup Detected: RSI={curr['rsi']:.2f}, Green={is_green}, Hook={rsi_hook}, "
                                f"HigherClose={higher_close}"
                            )
                            reason_components.append("LONG setup: RSI<oversold & band_touch & anti-knife ok")
                            # --- 1m Confirmation Check ---
                            confirmed_1m = False
                            if confirm_df is not None and not confirm_df.empty:
                                # Check last 2 candles of 1m data
                                last_1m = confirm_df.iloc[-2:]
                                touched_bb = last_1m["low"] <= last_1m.get("bb_lower", last_1m["low"])
                                touched_kc = last_1m["low"] <= last_1m.get("kc_lower", last_1m["low"])
                                confirmed_1m = bool((touched_bb | touched_kc).any())
                                if not confirmed_1m:
                                    logger.debug("LONG Signal ignored: No 1m confirmation (Low didn't touch bands)")
                                    reason_components.append("LONG filtered: no 1m confirmation")
                            else:
                                logger.warning("No confirmation data provided for LONG signal check.")
                                confirmed_1m = True
                                reason_components.append("No 1m data: bypassing 1m confirmation")

                            # --- Level 2 Confirmation (OBI) ---
                            obi_filter = True  # Pass by default
                            if getattr(Config, "USE_L2_FILTERS", False) and confirmed_1m and l2_features:
                                obi = l2_features.get("obi", 0.0)
                                obi_threshold = getattr(Config, "OBI_LONG_THRESHOLD", -0.4)
                                # STANDARD LOGIC: Avoid catching falling knives (Extreme Negative OBI)
                                if obi < obi_threshold:  # Require OBI > Threshold (Support)
                                    logger.debug(
                                        f"LONG Signal ignored: OBI ({obi:.2f}) < {obi_threshold} "
                                        f"(Extreme Sell Pressure)"
                                    )
                                    obi_filter = False
                                    reason_components.append(f"LONG filtered: OBI ({obi:.2f}) < {obi_threshold}")

                        # --- Volume Filter ---
                        # DISABLED: Volume filter was too aggressive even at 0.5x MA
                        # We rely on OBI and Price Action for confirmation
                        vol_filter = True
                        # volume = curr.get("volume", 0)
                        # volume_ma = curr.get("volume_ma", 0)
                        # if volume < volume_ma * 0.5:
                        #     logger.debug(f"LONG Signal ignored: Low Volume ({volume:.2f} < {volume_ma:.2f})")
                        #     reason_components.append(f"LONG filtered: low volume")
                        #     vol_filter = False

                        if confirmed_1m and obi_filter and vol_filter:
                            # Check if we already reverted to mean in the same candle
                            bb_mid = curr.get("bb_mid", curr["close"])
                            if curr["close"] >= bb_mid:
                                logger.debug("LONG Signal ignored: Price already reverted to mean (Close >= BB_Mid)")
                                reason_components.append("LONG filtered: already at/above BB_Mid")
                                signal = None
                            else:
                                # Check Minimum Potential Profit (Distance to Mean)
                                potential_profit = (bb_mid - curr["close"]) / curr["close"]
                                min_profit = getattr(Config, "MEAN_REV_MIN_PROFIT", 0.02)

                                if potential_profit < min_profit:
                                    logger.debug(
                                        f"LONG Signal ignored: Insufficient Potential Profit "
                                        f"({potential_profit:.2%}) < {min_profit:.1%}"
                                    )
                                    reason_components.append(
                                        f"LONG filtered: insufficient potential profit "
                                        f"({potential_profit:.2%} < {min_profit:.1%})"
                                    )
                                    signal = None
                                else:
                                    signal = "LONG"
                                    pre_ml_signal = "LONG"
                                    reason_components.append(
                                        f"LONG candidate: potential_profit={potential_profit:.2%}>= {min_profit:.1%}"
                                    )
                                    # Dynamic sizing: deeper RSI = larger position
                                    rsi_depth = (Config.RSI_OVERSOLD - curr["rsi"]) / 10  # 0-3 range
                                    size_multiplier = min(1.0 + rsi_depth, 2.0) * onchain_multiplier
                        else:
                            logger.debug(
                                f"LONG Signal ignored: No Anti-Knife Confirmation (Green={is_green}, Hook={rsi_hook})"
                            )
                            reason_components.append("LONG filtered: anti-knife failed (green & rsi_hook required)")

            # SHORT: Downtrend + Touch Upper Bands (BB OR KC) + RSI Overbought
            elif is_downtrend_1h and curr["rsi"] > Config.RSI_OVERBOUGHT:
                # Check if price touches either Bollinger Upper OR Keltner Upper (Current OR Previous)
                band_buffer = getattr(Config, "BAND_ATR_BUFFER", 0.0) * curr.get("atr", 0.0)
                touched_band = (
                    (curr["high"] >= curr["bb_upper"] + band_buffer)
                    or (curr["high"] >= curr.get("kc_upper", curr["bb_upper"]) + band_buffer)
                    or (prev["high"] >= prev["bb_upper"] + band_buffer)
                    or (prev["high"] >= prev.get("kc_upper", prev["bb_upper"]) + band_buffer)
                )

                if touched_band:
                    # --- Anti-Knife Confirmation ---
                    is_red = curr["close"] < curr["open"]
                    rsi_hook = curr["rsi"] < prev["rsi"]
                    lower_close = curr["close"] < prev["close"]

                    if is_red and rsi_hook and lower_close:
                        reason_components.append("SHORT setup: RSI>overbought & band_touch & anti-knife ok")
                        # --- 1m Confirmation Check ---
                        confirmed_1m = False
                        if confirm_df is not None and not confirm_df.empty:
                            # Check last 2 candles of 1m data
                            last_1m = confirm_df.iloc[-2:]
                            touched_bb = last_1m["high"] >= last_1m.get("bb_upper", last_1m["high"])
                            touched_kc = last_1m["high"] >= last_1m.get("kc_upper", last_1m["high"])
                            confirmed_1m = bool((touched_bb | touched_kc).any())
                            if not confirmed_1m:
                                logger.debug("SHORT Signal ignored: No 1m confirmation (High didn't touch bands)")
                                reason_components.append("SHORT filtered: no 1m confirmation")
                        else:
                            logger.warning("No confirmation data provided for SHORT signal check.")
                            confirmed_1m = True
                            reason_components.append("No 1m data: bypassing 1m confirmation")

                        # --- Level 2 Confirmation (OBI) ---
                        obi_filter = True  # Pass by default
                        if getattr(Config, "USE_L2_FILTERS", False) and confirmed_1m and l2_features:
                            obi = l2_features.get("obi", 0.0)
                            obi_threshold = getattr(Config, "OBI_SHORT_THRESHOLD", 0.4)
                            # STANDARD LOGIC: Avoid shorting into pumps (Extreme Positive OBI)
                            if obi > obi_threshold:  # Require OBI < Threshold (Resistance)
                                logger.debug(
                                    f"SHORT Signal ignored: OBI ({obi:.2f}) > {obi_threshold} (Extreme Buy Pressure)"
                                )
                                obi_filter = False
                                confirmed_1m = False
                                reason_components.append(f"SHORT filtered: OBI ({obi:.2f}) > {obi_threshold}")

                        # --- Volume Filter ---
                        # DISABLED: Volume filter was too aggressive even at 0.5x MA
                        # We rely on OBI and Price Action for confirmation
                        vol_filter = True
                        # volume = curr.get("volume", 0)
                        # volume_ma = curr.get("volume_ma", 0)
                        # if volume < volume_ma * 0.5:
                        #      logger.debug(f"SHORT Signal ignored: Low Volume ({volume:.2f} < {volume_ma:.2f})")
                        #      reason_components.append(f"SHORT filtered: low volume")
                        #      vol_filter = False

                        if confirmed_1m and obi_filter and vol_filter:
                            # Check if we already reverted to mean in the same candle
                            if curr["close"] <= curr.get("bb_mid", curr["close"]):
                                logger.debug("SHORT Signal ignored: Price already reverted to mean (Close <= BB_Mid)")
                                reason_components.append("SHORT filtered: already at/below BB_Mid")
                                signal = None
                            else:
                                signal = "SHORT"
                                pre_ml_signal = "SHORT"
                                # Dynamic sizing: higher RSI = larger position
                                rsi_depth = (curr["rsi"] - Config.RSI_OVERBOUGHT) / 10  # 0-3 range
                                size_multiplier = min(1.0 + rsi_depth, 2.0) * onchain_multiplier
                    else:
                        logger.debug("SHORT Signal ignored: No Anti-Knife Confirmation (Red Candle + RSI Hook)")
                        reason_components.append("SHORT filtered: anti-knife failed (red & rsi_hook required)")

            # --- 5. ML Filter with Tighter Thresholds & Confidence Weighting ---
            ml_filter = True  # Pass by default
            if Config.ML_ENABLED and ml_score is not None:
                # Apply ML filter
                if signal == "LONG":
                    # Tier 1: High Confidence
                    if ml_score >= Config.ML_LONG_THRESHOLD:
                        reason_components.append(f"LONG ML Tier 1 (score={ml_score:.2f} >= {Config.ML_LONG_THRESHOLD})")

                    # Tier 2: Medium Confidence (Reduced Size)
                    elif ml_score >= Config.ML_LONG_THRESHOLD_LOW:
                        size_multiplier *= 0.5
                        reason_components.append(
                            f"LONG ML Tier 2 (score={ml_score:.2f} >= {Config.ML_LONG_THRESHOLD_LOW}): Size x0.5"
                        )

                    else:
                        logger.debug(f"LONG Signal Filtered by ML: {ml_score:.2f} < {Config.ML_LONG_THRESHOLD_LOW}")
                        reason_components.append(
                            f"LONG filtered by ML: score={ml_score:.2f} < {Config.ML_LONG_THRESHOLD_LOW}"
                        )
                        signal = None
                        size_multiplier = 0.0
                        ml_filter = False

                    # Super High Confidence Boost
                    if signal and ml_score > 0.8:
                        size_multiplier *= 1.5
                        reason_components.append(f"LONG ML high confidence boost (score={ml_score:.2f})")
                        logger.info(f"High Confidence LONG (Score={ml_score:.2f}): Boosted Size x1.5")

                elif signal == "SHORT":
                    # Tier 1: High Confidence
                    if ml_score <= Config.ML_SHORT_THRESHOLD:
                        reason_components.append(
                            f"SHORT ML Tier 1 (score={ml_score:.2f} <= {Config.ML_SHORT_THRESHOLD})"
                        )

                    # Tier 2: Medium Confidence (Reduced Size)
                    elif ml_score <= Config.ML_SHORT_THRESHOLD_LOW:
                        size_multiplier *= 0.5
                        reason_components.append(
                            f"SHORT ML Tier 2 (score={ml_score:.2f} <= {Config.ML_SHORT_THRESHOLD_LOW}): Size x0.5"
                        )

                    else:
                        logger.debug(f"SHORT Signal Filtered by ML: {ml_score:.2f} > {Config.ML_SHORT_THRESHOLD_LOW}")
                        reason_components.append(
                            f"SHORT filtered by ML: score={ml_score:.2f} > {Config.ML_SHORT_THRESHOLD_LOW}"
                        )
                        signal = None
                        size_multiplier = 0.0
                        ml_filter = False

                    # Super High Confidence Boost
                    if signal and ml_score < 0.2:
                        size_multiplier *= 1.5
                        reason_components.append(f"SHORT ML high confidence boost (score={ml_score:.2f})")
                        logger.info(f"High Confidence SHORT (Score={ml_score:.2f}): Boosted Size x1.5")

            elif Config.ML_ENABLED and ml_score is None:
                logger.debug("ML Score is None (Neutral/Offline). Filtering signal.")
                reason_components.append("ML score None: filtered (strict mode)")
                signal = None
                size_multiplier = 0.0
                ml_filter = False

            elif not Config.ML_ENABLED:
                # Explicitly log that ML is disabled
                reason_components.append("ML disabled: TA-only mode")

        side: Optional[str] = None
        if signal == "LONG":
            side = "LONG"
        elif signal == "SHORT":
            side = "SHORT"

        reason_string = " & ".join(reason_components) if reason_components else ""

        decision_context = {
            "timestamp": curr.name,
            "open": curr["open"],
            "high": curr["high"],
            "low": curr["low"],
            "close": curr["close"],
            "volume": curr.get("volume"),
            "volume_ma": curr.get("volume_ma"),
            "onchain_activity": onchain_activity,
            "rsi": curr["rsi"],
            "bb_lower": curr.get("bb_lower"),
            "bb_mid": curr.get("bb_mid", curr["close"]),
            "bb_upper": curr.get("bb_upper"),
            "kc_lower": curr.get("kc_lower"),
            "kc_upper": curr.get("kc_upper"),
            "ema200": curr.get("ema200"),
            "ema200_1h": ema200_1h,
            "atr": curr.get("atr", 0.0),
            "adx": adx,
            "regime": regime,
            "is_ranging": is_valid_regime,  # Reusing key for backward compatibility
            "is_uptrend_1h": is_uptrend_1h,
            "is_downtrend_1h": is_downtrend_1h,
            "touched_band": touched_band,
            "is_green": is_green,  # or is_red for short, simplified
            "rsi_hook": rsi_hook,
            "confirmed_1m": confirmed_1m,
            "obi_filter": obi_filter,
            "l2_used": getattr(Config, "USE_L2_FILTERS", False),
            "obi": obi,
            "spread": spread,
            "market_depth_ratio": market_depth_ratio,
            "ml_score": ml_score,
            "ml_filter": ml_filter,
            "pre_ml_signal": pre_ml_signal,
            "final_signal": signal,
            "side": side,
            "reason_string": reason_string,
        }

        base_result.update({"signal": signal, "size_multiplier": size_multiplier, "decision_context": decision_context})

        return base_result

    def calculate_position_size(self, balance: float, entry_price: float, atr: float, multiplier: float = 1.0) -> float:
        """
        Calculates position size based on risk and ML multiplier.
        Caps size at available balance (Spot limit).
        """
        if multiplier == 0.0:
            return 0.0

        risk_amount = balance * Config.RISK_PER_TRADE * multiplier
        sl_distance = Config.ATR_MULTIPLIER * atr

        if sl_distance == 0:
            return 0.0

        qty = risk_amount / sl_distance

        # Cap quantity at available balance (Spot Limit)
        # Use 0.99 factor to prevent rounding errors causing insufficient funds
        max_qty = (balance * 0.99) / entry_price

        if qty > max_qty:
            qty = max_qty

        return qty

    def get_exit_updates(self, position: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determines exit updates with:
        1. Reversion to Mean (Middle Bollinger Band)
        2. MIN_ROI Target Achieved (0.7%)
        3. Ratcheting Stop Loss (3-Stage Profit Locking)

        Ratcheting Stages:
        - Stage 1 (0.5% ROI): Move SL to breakeven
        - Stage 2 (1.0% ROI): Lock in 0.5% profit
        - Stage 3 (1.5% ROI): Dynamic trailing by 1.5x ATR
        """
        updates: Dict[str, Any] = {}
        side = position["side"]
        entry_price = position["entry_price"]
        current_price = analysis["close"]
        bb_mid = analysis.get("bb_mid", current_price)  # Safe access
        atr = analysis.get("atr", 0.0)
        current_sl = position.get("stop_loss", 0.0)

        # Calculate current profit percentage
        if side == "LONG":
            profit_pct = (current_price - entry_price) / entry_price
        else:  # SHORT
            profit_pct = (entry_price - current_price) / entry_price

        should_exit = False

        # Exit Condition 1: Reversion to Mean (Middle Bollinger Band)
        # Only exit if we have at least 0.5% profit to cover fees and slippage
        min_exit_roi = 0.005
        if side == "LONG" and current_price >= bb_mid:
            if profit_pct > min_exit_roi:
                should_exit = True
                logger.info(
                    f"LONG Exit: Price reverted to mean ({current_price:.5f} >= {bb_mid:.5f}) & "
                    f"ROI > {min_exit_roi:.1%}"
                )
            else:
                logger.debug(f"LONG Mean Reversion reached but ROI ({profit_pct:.2%}) < {min_exit_roi:.1%}. Holding.")
        elif side == "SHORT" and current_price <= bb_mid:
            if profit_pct > min_exit_roi:
                should_exit = True
                logger.info(
                    f"SHORT Exit: Price reverted to mean ({current_price:.5f} <= {bb_mid:.5f}) & "
                    f"ROI > {min_exit_roi:.1%}"
                )
            else:
                logger.debug(f"SHORT Mean Reversion reached but ROI ({profit_pct:.2%}) < {min_exit_roi:.1%}. Holding.")

        # Exit Condition 2: MIN_ROI Target Achieved
        # REMOVED: We now rely on Ratcheting Stop Loss to let winners run.
        # This allows us to capture larger moves if the reversion turns into a trend continuation.

        # Explicit Exit Signal
        if should_exit:
            updates["exit_signal"] = True
            updates["exit_reason"] = "Mean Reversion"
            return updates

        # --- Ratcheting Stop Loss Logic ---
        new_sl = None

        if side == "LONG":
            # Stage 3: Dynamic Trailing
            if profit_pct > Config.RATCHET_TRAIL_ROI:
                trail_sl = current_price - (atr * Config.RATCHET_TRAIL_ATR_MULTIPLIER)
                if trail_sl > current_sl:
                    new_sl = trail_sl
                    logger.debug(f"Ratcheting SL (Stage 3): Trailing to {new_sl:.5f}")

            # Stage 2: Lock Profit
            elif profit_pct > Config.RATCHET_LOCK_PROFIT_ROI:
                lock_sl = entry_price * 1.005  # Lock 0.5% profit
                if lock_sl > current_sl:
                    new_sl = lock_sl
                    logger.debug(f"Ratcheting SL (Stage 2): Locking 0.5% profit at {new_sl:.5f}")

            # Stage 1: Breakeven
            elif profit_pct > Config.RATCHET_BREAKEVEN_ROI:
                if current_sl < entry_price:
                    new_sl = entry_price
                    logger.debug(f"Ratcheting SL (Stage 1): Moving to Breakeven at {new_sl:.5f}")

        else:  # SHORT
            # Stage 3: Dynamic Trailing
            if profit_pct > Config.RATCHET_TRAIL_ROI:
                trail_sl = current_price + (atr * Config.RATCHET_TRAIL_ATR_MULTIPLIER)
                if trail_sl < current_sl or current_sl == 0:
                    new_sl = trail_sl
                    logger.debug(f"Ratcheting SL (Stage 3): Trailing to {new_sl:.5f}")

            # Stage 2: Lock Profit
            elif profit_pct > Config.RATCHET_LOCK_PROFIT_ROI:
                lock_sl = entry_price * 0.995  # Lock 0.5% profit
                if lock_sl < current_sl or current_sl == 0:
                    new_sl = lock_sl
                    logger.debug(f"Ratcheting SL (Stage 2): Locking 0.5% profit at {new_sl:.5f}")

            # Stage 1: Breakeven
            elif profit_pct > Config.RATCHET_BREAKEVEN_ROI:
                if current_sl > entry_price or current_sl == 0:
                    new_sl = entry_price
                    logger.debug(f"Ratcheting SL (Stage 1): Moving to Breakeven at {new_sl:.5f}")

        if new_sl:
            updates["new_sl"] = new_sl

        return updates
