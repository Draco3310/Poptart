import pandas as pd
import pytest

from src.config import Config
from src.strategies.mean_reversion_mtf import MeanReversionStrategy as Strategy


class TestStrategy:
    def test_long_entry(self, ohlcv_5m_fixture: pd.DataFrame, ohlcv_1m_fixture: pd.DataFrame) -> None:
        """
        Scenario 1: Mean Reversion Long Entry with 1m Confirmation

        Logic:
        - Uptrend (Close > EMA200_1h)
        - Dip (Low <= Lower BB OR KC)
        - Oversold (RSI < 30)
        - Confirmation: 1m Low touches Lower Band
        """
        df = ohlcv_5m_fixture.copy()
        last_idx = df.index[-1]

        # 1. Setup Indicators (Primary 5m)
        # Trend: Up (1H EMA)
        df.loc[last_idx, "ema200_1h"] = 100.0
        df.loc[last_idx, "ema200"] = 100.0

        # Price: Dip (low touches lower band)
        df.loc[last_idx, "close"] = 106.0
        df.loc[last_idx, "open"] = 105.5  # Green Candle (Close > Open)
        df.loc[last_idx, "low"] = 105.0  # Touch Lower Band
        df.loc[last_idx, "high"] = 107.0

        # Bollinger Bands
        df.loc[last_idx, "bb_lower"] = 105.0  # Touch Lower Band
        df.loc[last_idx, "bb_upper"] = 115.0
        df.loc[last_idx, "bb_mid"] = 110.0

        # Keltner Channels
        df.loc[last_idx, "kc_lower"] = 104.0
        df.loc[last_idx, "kc_upper"] = 116.0
        df.loc[last_idx, "kc_mid"] = 110.0

        # RSI: Oversold
        df.loc[last_idx, "rsi"] = 24.0  # Must be < 25 (Tightened)
        # RSI Hook (Current > Previous)
        prev_idx = df.index[-2]
        df.loc[prev_idx, "rsi"] = 20.0
        df.loc[prev_idx, "close"] = 105.0  # Ensure Higher Close

        # ADX: Ranging
        df.loc[last_idx, "adx"] = 20.0  # Must be < 25

        # Volume indicators
        df.loc[last_idx, "volume_ma"] = 1000.0
        df.loc[last_idx, "onchain_activity"] = 1000.0

        # Other required fields
        df.loc[last_idx, "atr"] = 0.2  # Low Volatility

        # 2. Setup Confirmation (1m)
        confirm_df = ohlcv_1m_fixture.copy()
        last_1m_idx = confirm_df.index[-1]

        # Make sure 1m touches lower band
        confirm_df.loc[last_1m_idx, "low"] = 105.0
        confirm_df.loc[last_1m_idx, "bb_lower"] = 105.0
        confirm_df.loc[last_1m_idx, "kc_lower"] = 104.0

        # 3. Execute
        strategy = Strategy()
        result = strategy.analyze(df, ml_score=0.9, confirm_df=confirm_df)

        # 4. Assert
        assert result["signal"] == "LONG"
        assert result["size_multiplier"] > 0.0

    def test_long_entry_no_confirmation(self, ohlcv_5m_fixture: pd.DataFrame, ohlcv_1m_fixture: pd.DataFrame) -> None:
        """
        Scenario 1b: Long Entry REJECTED due to missing 1m confirmation
        """
        df = ohlcv_5m_fixture.copy()
        last_idx = df.index[-1]

        # 1. Setup Indicators (Primary 5m) - Valid Signal
        df.loc[last_idx, "ema200_1h"] = 100.0
        df.loc[last_idx, "ema200"] = 100.0
        df.loc[last_idx, "close"] = 106.0
        df.loc[last_idx, "low"] = 105.0
        df.loc[last_idx, "high"] = 107.0
        df.loc[last_idx, "bb_lower"] = 105.0
        df.loc[last_idx, "bb_upper"] = 115.0
        df.loc[last_idx, "bb_mid"] = 110.0
        df.loc[last_idx, "kc_lower"] = 104.0
        df.loc[last_idx, "kc_upper"] = 116.0
        df.loc[last_idx, "rsi"] = 24.0
        df.loc[last_idx, "adx"] = 20.0
        df.loc[last_idx, "volume_ma"] = 1000.0
        df.loc[last_idx, "onchain_activity"] = 1000.0
        df.loc[last_idx, "atr"] = 0.2

        # 2. Setup Confirmation (1m) - NO TOUCH
        confirm_df = ohlcv_1m_fixture.copy()
        last_1m_idx = confirm_df.index[-1]

        # 1m Low is ABOVE bands
        confirm_df.loc[last_1m_idx, "low"] = 106.0
        confirm_df.loc[last_1m_idx, "bb_lower"] = 105.0
        confirm_df.loc[last_1m_idx, "kc_lower"] = 104.0

        # 3. Execute
        strategy = Strategy()
        result = strategy.analyze(df, ml_score=0.9, confirm_df=confirm_df)

        # 4. Assert
        assert result["signal"] is None

    def test_short_entry(self, ohlcv_5m_fixture: pd.DataFrame, ohlcv_1m_fixture: pd.DataFrame) -> None:
        """
        Scenario 2: Mean Reversion Short Entry with 1m Confirmation

        Logic:
        - Downtrend (Close < EMA200_1h)
        - Rally (High >= Upper BB OR KC)
        - Overbought (RSI > 70)
        - Confirmation: 1m High touches Upper Band
        """
        df = ohlcv_5m_fixture.copy()
        last_idx = df.index[-1]

        # 1. Setup Indicators (Primary 5m)
        # Trend: Down (1H EMA)
        df.loc[last_idx, "ema200_1h"] = 120.0
        df.loc[last_idx, "ema200"] = 120.0

        # Price: Rally (high touches upper band)
        df.loc[last_idx, "close"] = 114.0
        df.loc[last_idx, "open"] = 114.5  # Red Candle (Close < Open)
        df.loc[last_idx, "high"] = 115.0  # Touch Upper Band
        df.loc[last_idx, "low"] = 113.0

        # Bollinger Bands
        df.loc[last_idx, "bb_lower"] = 105.0
        df.loc[last_idx, "bb_upper"] = 115.0  # Touch Upper Band
        df.loc[last_idx, "bb_mid"] = 110.0

        # Keltner Channels
        df.loc[last_idx, "kc_lower"] = 104.0
        df.loc[last_idx, "kc_upper"] = 116.0
        df.loc[last_idx, "kc_mid"] = 110.0

        # RSI: Overbought
        df.loc[last_idx, "rsi"] = 76.0  # Must be > 75 (Tightened)
        # RSI Hook (Current < Previous)
        prev_idx = df.index[-2]
        df.loc[prev_idx, "rsi"] = 80.0
        df.loc[prev_idx, "close"] = 115.0  # Ensure Lower Close

        # ADX: Ranging
        df.loc[last_idx, "adx"] = 20.0  # Must be < 25

        # Volume indicators
        df.loc[last_idx, "volume_ma"] = 1000.0
        df.loc[last_idx, "onchain_activity"] = 1000.0

        # Other required fields
        df.loc[last_idx, "atr"] = 0.2

        # 2. Setup Confirmation (1m)
        confirm_df = ohlcv_1m_fixture.copy()
        last_1m_idx = confirm_df.index[-1]

        # Make sure 1m touches upper band
        confirm_df.loc[last_1m_idx, "high"] = 115.0
        confirm_df.loc[last_1m_idx, "bb_upper"] = 115.0
        confirm_df.loc[last_1m_idx, "kc_upper"] = 116.0

        # 3. Execute
        strategy = Strategy()
        # Short needs Low Score (< 0.4)
        result = strategy.analyze(df, ml_score=0.2, confirm_df=confirm_df)

        # 4. Assert
        assert result["signal"] == "SHORT"
        assert result["size_multiplier"] > 0.0

    def test_l2_confirmation(self, ohlcv_5m_fixture: pd.DataFrame) -> None:
        """
        Scenario 6: Level 2 Confirmation (OBI)
        """
        df = ohlcv_5m_fixture.copy()
        last_idx = df.index[-1]

        # Setup Valid Long Signal
        df.loc[last_idx, "ema200_1h"] = 100.0
        df.loc[last_idx, "ema200"] = 100.0
        df.loc[last_idx, "close"] = 106.0
        df.loc[last_idx, "open"] = 105.5
        df.loc[last_idx, "low"] = 105.0
        df.loc[last_idx, "high"] = 107.0
        df.loc[last_idx, "bb_lower"] = 105.0
        df.loc[last_idx, "bb_upper"] = 115.0
        df.loc[last_idx, "bb_mid"] = 110.0
        df.loc[last_idx, "kc_lower"] = 104.0
        df.loc[last_idx, "kc_upper"] = 116.0
        df.loc[last_idx, "rsi"] = 24.0
        df.loc[last_idx, "adx"] = 20.0
        df.loc[last_idx, "atr"] = 0.2
        df.loc[last_idx, "volume_ma"] = 1000.0
        df.loc[last_idx, "onchain_activity"] = 1000.0

        # RSI Hook
        prev_idx = df.index[-2]
        df.loc[prev_idx, "rsi"] = 20.0
        df.loc[prev_idx, "close"] = 105.0  # Ensure Higher Close

        strategy = Strategy()

        # Case A: Negative OBI (Should Reject Long)
        l2_bad = {"obi": -0.5, "spread": 0.0001, "market_depth_ratio": 0.5}
        result_bad = strategy.analyze(df, ml_score=0.9, l2_features=l2_bad)
        assert result_bad["signal"] is None

        # Case B: Positive OBI (Should Accept Long)
        l2_good = {"obi": 0.5, "spread": 0.0001, "market_depth_ratio": 2.0}
        result_good = strategy.analyze(df, ml_score=0.9, l2_features=l2_good)
        assert result_good["signal"] == "LONG"

    def test_ml_filter_logic(self, ohlcv_fixture: pd.DataFrame) -> None:
        """
        Scenario 3: ML Filter Logic
        """
        df = ohlcv_fixture.copy()
        last_idx = df.index[-1]

        # Setup Valid Long Signal
        df.loc[last_idx, "ema200_1h"] = 100.0
        df.loc[last_idx, "ema200"] = 100.0
        df.loc[last_idx, "close"] = 106.0
        df.loc[last_idx, "open"] = 105.5  # Green Candle
        df.loc[last_idx, "low"] = 105.0
        df.loc[last_idx, "high"] = 107.0
        df.loc[last_idx, "bb_lower"] = 105.0
        df.loc[last_idx, "bb_upper"] = 115.0
        df.loc[last_idx, "bb_mid"] = 110.0
        df.loc[last_idx, "kc_lower"] = 104.0
        df.loc[last_idx, "kc_upper"] = 116.0
        df.loc[last_idx, "rsi"] = 24.0  # Must be < 25

        # RSI Hook
        prev_idx = df.index[-2]
        df.loc[prev_idx, "rsi"] = 20.0

        df.loc[last_idx, "adx"] = 20.0  # Must be < 25
        df.loc[last_idx, "atr"] = 0.2
        df.loc[last_idx, "volume_ma"] = 1000.0
        df.loc[last_idx, "onchain_activity"] = 1000.0

        strategy = Strategy()

        # Enable ML
        original_ml = Config.ML_ENABLED
        Config.ML_ENABLED = True

        try:
            # Test Low Score (Should Filter Long)
            result = strategy.analyze(df, ml_score=0.4)  # < 0.6 Threshold
            assert result["signal"] is None

        finally:
            Config.ML_ENABLED = original_ml

    def test_exit_logic(self, ohlcv_fixture: pd.DataFrame) -> None:
        """
        Scenario 4: Exit at Mean
        """
        strategy = Strategy()

        # Mock Position
        position = {"side": "LONG", "entry_price": 100.0, "stop_loss": 90.0, "tp1_hit": False}

        # Mock Analysis (Price at Mean)
        analysis = {
            "close": 110.0,
            "bb_mid": 110.0,
            "rsi": 50.0,
            "atr": 1.0,
            "donchian_high": 120.0,
            "donchian_low": 90.0,
        }

        updates = strategy.get_exit_updates(position, analysis)

        # Should trigger exit signal
        assert updates.get("exit_signal") is True
        assert updates.get("exit_reason") == "Mean Reversion"

    def test_ratchet_stop_loss(self, ohlcv_fixture: pd.DataFrame) -> None:
        """
        Scenario 5: Ratcheting Stop Loss (Stage 2: Lock Profit)
        """
        strategy = Strategy()

        # Mock Position (Long from 100)
        position = {"side": "LONG", "entry_price": 100.0, "stop_loss": 99.0}

        # Mock Analysis (Price at 103.0 - 3.0% Profit)
        # Should trigger Stage 2 (Lock 0.5% Profit -> SL = 100.5)
        analysis = {
            "close": 103.0,
            "bb_mid": 105.0,  # Not at mean yet
            "atr": 1.0,
            "rsi": 60.0,
        }

        updates = strategy.get_exit_updates(position, analysis)

        assert "new_sl" in updates
        assert updates["new_sl"] == pytest.approx(101.0)  # Stage 3 Trailing (103 - 2*1 = 101)
