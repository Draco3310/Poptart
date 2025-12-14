import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from src.predictors.base_predictor import BasePredictor
from src.predictors.lstm_model import LSTMPredictor
from src.predictors.random_forest import RandomForestPredictor
from src.predictors.xgboost_model import XGBoostPredictor

logger = logging.getLogger(__name__)


class PredictorOrchestrator:
    """
    Manages multiple predictors and calculates an ensemble score.

    Current Configuration (Phase 2B):
    - Models: Random Forest & XGBoost.
    - Target: TP-before-SL over 1 hour (TP/SL H1).
    - Features: Volume Profile v2 (Daily POC/VAH/VAL) + Microstructure.
    """

    def __init__(self) -> None:
        self.predictors: List[BasePredictor] = []
        self.weights: List[float] = []

    def load_predictors(self, config_predictors: List[Dict[str, Any]]) -> None:
        """
        Loads predictors based on configuration.

        Args:
            config_predictors: List of dicts, e.g.:
            [
                {'type': 'xgboost', 'path': 'models/xgb_v1.json', 'weight': 0.4},
                {'type': 'lstm', 'path': 'models/lstm_v1.h5', 'weight': 0.4},
                {'type': 'rf', 'path': 'models/rf_v1.joblib', 'weight': 0.2}
            ]
        """
        for cfg in config_predictors:
            p_type = cfg.get("type")
            path = cfg.get("path")
            weight = cfg.get("weight", 1.0)

            predictor: Optional[BasePredictor] = None
            try:
                if p_type == "xgboost" or p_type == "xgb":
                    predictor = XGBoostPredictor()
                elif p_type == "lstm":
                    predictor = LSTMPredictor()
                elif p_type == "rf":
                    predictor = RandomForestPredictor()
                else:
                    logger.warning(f"Unknown predictor type: {p_type}")
                    continue

                if predictor and path:
                    predictor.load_model(path)
                    self.predictors.append(predictor)
                    self.weights.append(weight)
                    logger.info(f"Loaded {p_type} predictor from {path} with weight {weight}")

            except Exception as e:
                logger.error(f"Failed to load predictor {p_type} from {path}: {e}")

    def get_ensemble_score(self, enriched_df: pd.DataFrame, regime: str = "UNKNOWN") -> Optional[float]:
        """
        Runs all predictors and returns a weighted average score.
        Adapts weights based on Market Regime.
        Returns None if no predictors are active or if the score is neutral (weak confidence).
        """
        if not self.predictors:
            # No predictors loaded -> No opinion
            return None

        scores = []
        valid_weights = []

        for i, predictor in enumerate(self.predictors):
            try:
                score = predictor.predict(enriched_df)
                scores.append(score)

                # Dynamic Weighting based on Regime
                weight = self.weights[i]

                # Example: Boost XGBoost in Trend, RF in Chop
                # This assumes we know the model type or name.
                # For now, we keep it simple or rely on the config weights.
                # Future: Check predictor.name or type

                valid_weights.append(weight)
                logger.debug(f"Predictor {i} score: {score}")
            except Exception as e:
                logger.error(f"Predictor {i} failed during inference: {e}")

        if not scores:
            return None

        # Weighted Average
        total_weight = sum(valid_weights)
        if total_weight == 0:
            return sum(scores) / len(scores)

        weighted_score = sum(s * w for s, w in zip(scores, valid_weights)) / total_weight

        # REMOVED: Confidence Threshold logic that returned None for neutral scores.
        # We now return the raw score so the Strategy can apply its own thresholds.
        # Returning None caused Strict ML Mode to block valid trades.

        return weighted_score
