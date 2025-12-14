import pandas as pd

from src.strategies.strategy_selector import StrategySelector


class TestStrategySelector:
    def test_regime_detection_range(self):
        selector = StrategySelector()

        # Create data that looks like Range (Low ADX)
        df = pd.DataFrame(
            {
                "close": [100] * 100,
                "adx": [15.0] * 100,  # Low ADX -> RANGE
                "hurst_exponent": [0.4] * 100,
                "autocorr_20": [-0.1] * 100,
                "volatility_ratio": [1.0] * 100,
                "vpin_proxy": [0.1] * 100,
                "atr": [1.0] * 100,
                "rsi": [50] * 100,
                "bb_lower": [90] * 100,
                "bb_upper": [110] * 100,
                "ema200": [100] * 100,
            }
        )

        result = selector.analyze(df)
        assert result["decision_context"]["regime"] == "RANGE"
        # Active strategy might be MeanReversion or None if no signal
        # But we check if it routed correctly.
        # Since we didn't set up a signal, active_strategy might be None or MeanReversion depending on implementation.
        # StrategySelector sets active_strategy if a strategy is executed.
        # In RANGE, it executes MeanReversion.
        assert result.get("active_strategy") == "MeanReversion"

    def test_regime_detection_trend(self):
        selector = StrategySelector()

        # Create data that looks like Trend (High ADX)
        df = pd.DataFrame(
            {
                "close": [100] * 100,
                "adx": [40.0] * 100,  # High ADX -> TREND
                "hurst_exponent": [0.6] * 100,
                "autocorr_20": [0.2] * 100,
                "volatility_ratio": [1.0] * 100,
                "vpin_proxy": [0.1] * 100,
                "atr": [1.0] * 100,
                "rsi": [50] * 100,
                "bb_lower": [90] * 100,
                "bb_upper": [110] * 100,
                "ema200": [100] * 100,
            }
        )
        # Set DatetimeIndex
        df.index = pd.date_range(start="2024-01-01", periods=100, freq="5min")

        result = selector.analyze(df)
        assert result["decision_context"]["regime"] == "TREND"
        assert result.get("active_strategy") == "TrendFollowing"
