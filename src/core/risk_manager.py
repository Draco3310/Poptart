import logging
from typing import Any, Dict, Optional

from src.config import Config

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Centralized Risk Management Module.
    Responsibilities:
    1. Position Sizing (Volatility Targeting)
    2. Circuit Breakers (Drawdown Protection)
    3. Trade Gating (Liquidity/Toxicity Checks)
    """

    def __init__(self) -> None:
        self.peak_balance = 0.0
        self.max_drawdown_limit = 0.05  # 5% Max Drawdown
        self.vol_target_annual = 0.40  # 40% Annualized Volatility Target
        self.trading_paused = False
        self.pause_until = None

    def check_circuit_breaker(self, current_balance: float) -> bool:
        """
        Checks if the circuit breaker should be triggered based on drawdown.
        Returns True if trading should be PAUSED.
        """
        if self.peak_balance == 0.0:
            self.peak_balance = current_balance
        else:
            self.peak_balance = max(self.peak_balance, current_balance)

        drawdown = (self.peak_balance - current_balance) / self.peak_balance

        if drawdown > self.max_drawdown_limit:
            logger.critical(f"🛑 CIRCUIT BREAKER TRIGGERED! Drawdown: {drawdown:.2%} > {self.max_drawdown_limit:.2%}")
            self.trading_paused = True
            return True

        return False

    def calculate_size(
        self, balance: float, entry_price: float, atr: float, multiplier: float = 1.0, regime: str = "CHOP"
    ) -> float:
        """Wrapper for backward compatibility."""
        result = self.calculate_size_with_meta(balance, entry_price, atr, multiplier, regime)
        return float(result["qty"])

    def calculate_size_with_meta(
        self, balance: float, entry_price: float, atr: float, multiplier: float = 1.0, regime: str = "CHOP"
    ) -> Dict[str, Any]:
        """
        Calculates position size and returns detailed diagnostics.
        """
        meta = {
            "qty": 0.0,
            "reason": "UNKNOWN",
            "circuit_breaker": self.trading_paused,
            "base_qty": 0.0,
            "max_vol_qty": 0.0,
            "max_spot_qty": 0.0,
            "regime_scale": 1.0,
        }

        if self.trading_paused:
            meta["reason"] = "CIRCUIT_BREAKER"
            return meta

        if atr == 0 or entry_price == 0:
            meta["reason"] = "ATR_OR_PRICE_ZERO"
            return meta

        # 1. Base Risk Sizing (Fixed % Risk)
        risk_amount = balance * Config.RISK_PER_TRADE * multiplier
        sl_distance = atr * Config.ATR_MULTIPLIER

        if sl_distance == 0:
            meta["reason"] = "SL_DISTANCE_ZERO"
            return meta

        base_qty = risk_amount / sl_distance
        meta["base_qty"] = base_qty

        # 2. Volatility Targeting Cap
        target_daily_vol = self.vol_target_annual / 16.0
        target_daily_risk_amt = balance * target_daily_vol
        max_qty_vol = target_daily_risk_amt / atr
        meta["max_vol_qty"] = max_qty_vol

        # 3. Regime Adjustment
        regime_scale = 1.0
        if regime == "TREND":
            regime_scale = 1.2
        elif regime == "CHOP":
            regime_scale = 0.8
        elif regime == "VOLATILITY":
            regime_scale = 0.0

        meta["regime_scale"] = regime_scale

        final_qty = min(base_qty, max_qty_vol) * regime_scale

        # 4. Spot Limit Cap
        max_spot_qty = (balance * 0.99) / entry_price
        meta["max_spot_qty"] = max_spot_qty

        final_qty = min(final_qty, max_spot_qty)
        meta["qty"] = final_qty

        if final_qty > 0:
            meta["reason"] = "OK"
        else:
            # Determine why it's zero
            if regime_scale == 0:
                meta["reason"] = "REGIME_VOLATILITY"
            elif max_spot_qty == 0:
                meta["reason"] = "SPOT_CAP_ZERO"
            elif max_qty_vol == 0:
                meta["reason"] = "VOL_CAP_ZERO"
            else:
                meta["reason"] = "CALC_ZERO"

        return meta

    def check_trade_gating(self, l2_features: Optional[Dict[str, float]]) -> bool:
        return bool(self.check_trade_gating_with_meta(l2_features)["allowed"])

    def check_trade_gating_with_meta(self, l2_features: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """
        Checks microstructure features for toxicity.
        Returns dict with 'allowed' and 'reason'.
        """
        result = {"allowed": True, "reason": "OK"}

        if not l2_features:
            return result

        # VPIN / Toxicity Check
        spread = l2_features.get("spread", 0.0)
        if spread > 0.005:  # 0.5% spread is too high for liquid pair
            result["allowed"] = False
            result["reason"] = "SPREAD_TOO_HIGH"
            return result

        return result
