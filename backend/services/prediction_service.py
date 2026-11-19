"""
Prediction Service - wraps the shared Scorer for the API
"""
from typing import Any, Dict, Tuple

from src.inference import Scorer


class PredictionService:
    def __init__(self):
        self.scorer = None
        try:
            self.scorer = Scorer()
            print(f"✓ Model loaded ({len(self.scorer.features)} features)")
        except Exception as e:
            print(f"⚠ Warning: could not load model files: {e}")

    @property
    def model(self):
        return self.scorer.model if self.scorer else None

    @property
    def approve_below(self):
        return self.scorer.approve_below

    @property
    def reject_from(self):
        return self.scorer.reject_from

    def build_frame(self, features: Dict[str, Any]):
        if self.scorer is None:
            raise RuntimeError("Model not loaded")
        return self.scorer.frame(features)

    def predict(self, features: Dict[str, Any]) -> Tuple[float, str]:
        if self.scorer is None:
            raise RuntimeError("Model not loaded")
        out = self.scorer.score(features)
        return out["probability"], out["decision"].upper()

    def get_feature_names(self):
        return self.scorer.features if self.scorer else []


_prediction_service = None


def get_prediction_service() -> PredictionService:
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service