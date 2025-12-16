import logging
from typing import Optional

from src.predictors.base_predictor import BasePredictor
from src.predictors.lstm_model import LSTMPredictor
from src.predictors.random_forest import RandomForestPredictor
from src.predictors.xgboost_model import XGBoostPredictor

logger = logging.getLogger(__name__)

class PredictorFactory:
    """
    Factory for creating predictor instances.
    Decouples instantiation from orchestration.
    """

    @staticmethod
    def create_predictor(predictor_type: str) -> Optional[BasePredictor]:
        """
        Creates a predictor instance based on type string.
        """
        p_type = predictor_type.lower()
        
        if p_type in ["xgboost", "xgb"]:
            return XGBoostPredictor()
        elif p_type == "lstm":
            return LSTMPredictor()
        elif p_type == "rf":
            return RandomForestPredictor()
        else:
            logger.warning(f"Unknown predictor type: {predictor_type}")
            return None
