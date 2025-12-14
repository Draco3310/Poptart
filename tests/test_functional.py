import numpy as np
import pandas as pd

from src.core.feature_engine import FeatureEngine


class TestFeatureEngine:
    """
    Functional tests for the FeatureEngine (The Brain).
    """

    def test_indicator_accuracy(self, ohlcv_fixture: pd.DataFrame) -> None:
        """
        Test Case 1: Indicator Accuracy

        Test Steps:
        1. Initialize FeatureEngine.
        2. Feed synthetic OHLCV data.
        3. Verify output DataFrame contains expected columns.
        4. Verify values are valid floats (no NaNs).

        Expected Results:
        - Columns 'rsi', 'macd', 'atr' exist.
        - No NaN values in the final DataFrame.
        """
        engine = FeatureEngine()
        enriched = engine.compute_features(ohlcv_fixture)

        # Check Columns
        assert "rsi" in enriched.columns
        assert "macd" in enriched.columns
        assert "atr" in enriched.columns

        # Check Types
        assert enriched["rsi"].dtype == float
        assert enriched["macd"].dtype == float

        # Check NaNs (Should be dropped by engine)
        assert not enriched.isnull().values.any()

        # Check Logic (RSI between 0 and 100)
        assert enriched["rsi"].min() >= 0
        assert enriched["rsi"].max() <= 100

    def test_data_cleaning(self, ohlcv_fixture: pd.DataFrame) -> None:
        """
        Test Case 2: Data Cleaning

        Test Steps:
        1. Inject NaN values into the input data.
        2. Feed dirty data to FeatureEngine.
        3. Verify engine handles it gracefully.

        Expected Results:
        - Engine does not crash.
        - NaNs are filled or rows dropped.
        """
        # Inject NaNs
        dirty_data = ohlcv_fixture.copy()
        dirty_data.loc[100:105, "close"] = np.nan

        engine = FeatureEngine()
        enriched = engine.compute_features(dirty_data)

        # Verify no NaNs remain
        assert not enriched.isnull().values.any()

        # Verify we still have data
        assert not enriched.empty
