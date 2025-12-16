import logging
from typing import Any, Dict

import pandas as pd

from src.config import PairConfig

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
        pair_config: PairConfig,
        dca_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyzes the market and portfolio state to decide on a Rebalancing Buy.

        Args:
            df: DataFrame with indicators.
            current_price: Current price of BTC.
            current_balance_btc: Current amount of BTC held.
            current_equity_usdt: Total portfolio value in USDT.
            pair_config: Pair-specific configuration.
            dca_config: Legacy configuration dict (target_allocation).

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
            target_allocation = pair_config.dca_target_allocation
            
            # Rebalance Buffer (5% relative)
            # Buy if allocation < 19% (for 20% target)
            lower_threshold = target_allocation * pair_config.dca_rebalance_buffer_lower

            if current_allocation >= lower_threshold:
                result["decision_context"]["reason"] = (
                    f"Allocation OK ({current_allocation:.2%} >= {lower_threshold:.2%})"
                )
                return result

            # 2. Signal Buy (Rebalance)
            # Calculate Shortfall
            target_value = current_equity_usdt * target_allocation
            shortfall_usdt = target_value - btc_value_usdt
            
            if shortfall_usdt < pair_config.dca_min_trade_amount: # Min trade size
                 result["decision_context"]["reason"] = f"Shortfall too small ({shortfall_usdt:.2f})"
                 return result

            result["signal"] = "LONG"
            result["size_multiplier"] = 1.0
            result["active_strategy"] = "BTCSmartDCA"
            result["decision_context"] = {
                "strategy": "BTCSmartDCA",
                "reason": (
                    f"Rebalance Buy (Alloc {current_allocation:.2%} < {lower_threshold:.2%})"
                ),
                "current_allocation": current_allocation,
                "current_balance_btc": current_balance_btc,
                "current_equity_usdt": current_equity_usdt,
                "target_allocation": target_allocation,
                "shortfall_usdt": shortfall_usdt
            }

            return result

        except Exception as e:
            logger.error(f"Error in BTCSmartDCAStrategy: {e}")
            return result

    def calculate_position_size(self, pair_config: PairConfig, entry_price: float, shortfall_usdt: float = 0.0) -> float:
        """
        Calculates the quantity of BTC to buy.
        If shortfall_usdt is provided, uses that (Rebalancing).
        Otherwise uses fixed notional (fallback).
        """
        notional = shortfall_usdt
        if notional <= 0:
            notional = float(pair_config.dca_notional_per_trade or 10.0)
            
        if entry_price > 0:
            return notional / entry_price
        return 0.0

    def get_exit_updates(self, position: Dict[str, Any], analysis: Dict[str, Any], pair_config: PairConfig) -> Dict[str, Any]:
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
            buffer = pair_config.dca_rebalance_buffer_upper

            if current_allocation > (target_allocation * buffer):
                # Calculate excess value
                target_value = current_equity_usdt * target_allocation
                excess_value = btc_value_usdt - target_value

                if excess_value > pair_config.dca_min_trade_amount:  # Min trade size
                    excess_qty = excess_value / current_price
                    updates["exit_qty"] = excess_qty
                    updates["exit_reason"] = f"Rebalance (Alloc {current_allocation:.2%} > {target_allocation:.2%})"

        except Exception as e:
            logger.error(f"Error in BTCSmartDCAStrategy.get_exit_updates: {e}")

        return updates
