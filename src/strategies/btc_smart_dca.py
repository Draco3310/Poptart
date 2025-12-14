import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


class BTCSmartDCAStrategy:
    """
    Smart DCA Strategy for BTC.

    Logic:
    1. Target Allocation: Aims to maintain a specific % of portfolio in BTC (e.g., 20%).
    2. Dip Buying: Only buys if RSI is below a threshold (e.g., 50).
    3. Long-Term Hold: Does not sell.

    This strategy bypasses the Regime Classifier and ML models.
    """

    def analyze(
        self,
        df: pd.DataFrame,
        current_price: float,
        current_balance_btc: float,
        current_equity_usdt: float,
        dca_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyzes the market and portfolio state to decide on a DCA buy.

        Args:
            df: DataFrame with indicators (must contain 'rsi').
            current_price: Current price of BTC.
            current_balance_btc: Current amount of BTC held.
            current_equity_usdt: Total portfolio value in USDT.
            dca_config: Configuration dict (target_allocation, dip_threshold, notional_per_trade).

        Returns:
            Dict with 'signal' ("LONG" or None), 'size_multiplier' (fixed), and 'decision_context'.
        """

        # Default Result
        result: Dict[str, Any] = {
            "signal": None,
            "size_multiplier": 0.0,
            "active_strategy": "BTCSmartDCA",
            "decision_context": {"strategy": "BTCSmartDCA", "reason": "No Signal"},
        }

        try:
            # 1. Check Allocation
            btc_value_usdt = current_balance_btc * current_price
            current_allocation = btc_value_usdt / current_equity_usdt if current_equity_usdt > 0 else 0.0
            target_allocation = dca_config.get("target_allocation", 0.20)

            if current_allocation >= target_allocation:
                result["decision_context"]["reason"] = (
                    f"Allocation Full ({current_allocation:.2%} >= {target_allocation:.2%})"
                )
                return result

            # 2. Check Dip Condition (RSI)
            # Get latest RSI
            if df.empty or "rsi" not in df.columns:
                result["decision_context"]["reason"] = "Missing Data/RSI"
                return result

            current_rsi = df.iloc[-1]["rsi"]
            dip_threshold = dca_config.get("dip_threshold_rsi", 50)

            if current_rsi >= dip_threshold:
                result["decision_context"]["reason"] = f"RSI too high ({current_rsi:.2f} >= {dip_threshold})"
                return result

            # 3. Signal Buy
            # We are below target allocation AND RSI is low.
            result["signal"] = "LONG"
            result["size_multiplier"] = 1.0  # Not used for fixed notional, but good practice
            result["active_strategy"] = "BTCSmartDCA"
            result["decision_context"] = {
                "strategy": "BTCSmartDCA",
                "reason": (
                    f"Dip Buy (Alloc {current_allocation:.2%} < {target_allocation:.2%}, "
                    f"RSI {current_rsi:.2f} < {dip_threshold})"
                ),
                "current_allocation": current_allocation,
                "current_rsi": current_rsi,
                "current_balance_btc": current_balance_btc,
                "current_equity_usdt": current_equity_usdt,
                "target_allocation": target_allocation,
            }

            return result

        except Exception as e:
            logger.error(f"Error in BTCSmartDCAStrategy: {e}")
            return result

    def calculate_position_size(self, dca_config: Dict[str, Any], entry_price: float) -> float:
        """
        Calculates the quantity of BTC to buy based on fixed notional amount.
        """
        notional = float(dca_config.get("notional_per_trade", 10.0))
        if entry_price > 0:
            return notional / entry_price
        return 0.0

    def get_exit_updates(self, position: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks for rebalancing (selling excess allocation).
        """
        updates: Dict[str, Any] = {}

        try:
            dc = analysis.get("decision_context", {})
            current_price = analysis.get("close", 0.0)

            if current_price <= 0:
                return updates

            # Retrieve portfolio state from context (passed from analyze)
            current_balance_btc = dc.get("current_balance_btc", 0.0)
            current_equity_usdt = dc.get("current_equity_usdt", 0.0)
            target_allocation = dc.get("target_allocation", 0.20)

            if current_equity_usdt <= 0:
                return updates

            # Calculate current allocation
            btc_value_usdt = current_balance_btc * current_price
            current_allocation = btc_value_usdt / current_equity_usdt

            # Rebalance Threshold (e.g. 5% relative buffer, so 20% -> 21%)
            # Or absolute buffer? Let's use relative 5% buffer.
            buffer = 1.05

            if current_allocation > (target_allocation * buffer):
                # Calculate excess value
                target_value = current_equity_usdt * target_allocation
                excess_value = btc_value_usdt - target_value

                if excess_value > 10.0:  # Min trade size
                    excess_qty = excess_value / current_price
                    updates["exit_qty"] = excess_qty
                    updates["exit_reason"] = f"Rebalance (Alloc {current_allocation:.2%} > {target_allocation:.2%})"

        except Exception as e:
            logger.error(f"Error in BTCSmartDCAStrategy.get_exit_updates: {e}")

        return updates
