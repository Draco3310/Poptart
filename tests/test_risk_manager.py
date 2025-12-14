from src.core.risk_manager import RiskManager


class TestRiskManager:
    def test_circuit_breaker(self):
        rm = RiskManager()
        rm.max_drawdown_limit = 0.10  # 10%

        # Peak = 1000
        assert not rm.check_circuit_breaker(1000)

        # Drawdown 5% (950)
        assert not rm.check_circuit_breaker(950)

        # Drawdown 11% (890)
        assert rm.check_circuit_breaker(890)
        assert rm.trading_paused

    def test_volatility_targeting(self):
        rm = RiskManager()
        rm.vol_target_annual = 0.40  # 40%

        balance = 10000
        entry_price = 100
        atr = 1.0  # 1% daily vol approx

        # Target Daily Risk = 10000 * (0.40 / 16) = 250
        # Max Qty Vol = 250 / 1.0 = 250 units

        # Base Risk (2%) = 200
        # Stop Distance (2.5 ATR from Config) = 2.5
        # Base Qty = 200 / 2.5 = 80.0 units

        # Should pick min(80.0, 250) = 80.0
        qty = rm.calculate_size(balance, entry_price, atr, multiplier=1.0, regime="CHOP")
        # Regime CHOP scales by 0.8 -> 64.0
        assert abs(qty - 64.0) < 0.1

    def test_regime_scaling(self):
        rm = RiskManager()
        balance = 10000
        entry_price = 100
        atr = 1.0

        # Base Qty = 80.0 (from above logic)

        # Trend -> 1.2x -> 96.0
        qty_trend = rm.calculate_size(balance, entry_price, atr, multiplier=1.0, regime="TREND")
        assert abs(qty_trend - 96.0) < 0.1

        # Volatility -> 0.0x -> 0
        qty_vol = rm.calculate_size(balance, entry_price, atr, multiplier=1.0, regime="VOLATILITY")
        assert qty_vol == 0.0
