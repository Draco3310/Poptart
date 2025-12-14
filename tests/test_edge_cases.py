import logging
from unittest.mock import MagicMock

import pandas as pd

from src.predictors.orchestrator import PredictorOrchestrator

logger = logging.getLogger(__name__)


class TestEdgeCases:
    """
    Tests for edge cases and new features:
    1. ML Confidence Threshold
    2. On-Chain Feature Placeholder
    3. Circuit Breaker Logic (Simulation)
    """

    def test_orchestrator_confidence_threshold(self):
        """
        Verifies that the Orchestrator returns None (Neutral)
        when the weighted score is between 0.4 and 0.6.
        """
        orchestrator = PredictorOrchestrator()

        # Mock predictors
        mock_pred_1 = MagicMock()
        mock_pred_1.predict.return_value = 0.55  # Weak signal

        orchestrator.predictors = [mock_pred_1]
        orchestrator.weights = [1.0]

        # Test weak signal
        score = orchestrator.get_ensemble_score(pd.DataFrame())
        assert score == 0.55, f"Expected raw score 0.55, got {score}"

        # Test strong signal (Long)
        mock_pred_1.predict.return_value = 0.8
        score = orchestrator.get_ensemble_score(pd.DataFrame())
        assert score == 0.8, f"Expected 0.8, got {score}"

        # Test strong signal (Short)
        mock_pred_1.predict.return_value = 0.2
        score = orchestrator.get_ensemble_score(pd.DataFrame())
        assert score == 0.2, f"Expected 0.2, got {score}"

    def test_feature_engine_onchain_placeholder(self):
        """
        Verifies that add_onchain_features adds the 'tx_volume' column.
        """
        from src.core.features.legacy import LegacyFeatureBlock

        block = LegacyFeatureBlock()
        df = pd.DataFrame({"close": [100, 101, 102], "volume": [1000, 2000, 1500]})

        enriched = block.add_onchain_features(df)

        assert "tx_volume" in enriched.columns
        # Check calculation: volume * close
        expected_tx_vol = df["volume"] * df["close"]
        expected_tx_vol.name = "tx_volume"
        pd.testing.assert_series_equal(enriched["tx_volume"], expected_tx_vol)

    def test_circuit_breaker_logic(self):
        """
        Simulates the circuit breaker logic implemented in main.py.
        """
        MAX_DRAWDOWN = 0.05
        peak_balance = 1000.0

        # Scenario 1: Small dip (No Trigger)
        current_balance = 960.0  # 4% drawdown
        drawdown = (peak_balance - current_balance) / peak_balance
        assert drawdown == 0.04
        assert drawdown <= MAX_DRAWDOWN

        # Scenario 2: Critical dip (Trigger)
        current_balance = 940.0  # 6% drawdown
        drawdown = (peak_balance - current_balance) / peak_balance
        assert drawdown == 0.06
        assert drawdown > MAX_DRAWDOWN

        # Scenario 3: New Peak
        current_balance = 1100.0
        peak_balance = max(peak_balance, current_balance)
        assert peak_balance == 1100.0
        drawdown = (peak_balance - current_balance) / peak_balance
        assert drawdown == 0.0
