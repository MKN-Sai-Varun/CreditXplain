"""
Counterfactual Service - bounded recourse search over applicant-controlled levers
"""
from typing import Any, Dict, List

from backend.services.prediction_service import get_prediction_service
from src.recourse import find_recourse


class CounterfactualService:
    def __init__(self):
        self.ps = get_prediction_service()

    def generate_counterfactuals(self, features: Dict[str, Any], num_counterfactuals: int = 3) -> List[Dict[str, Any]]:
        if self.ps.scorer is None:
            return []
        try:
            return find_recourse(self.ps.scorer, features, n=num_counterfactuals)
        except Exception as e:
            print(f"⚠ Counterfactual generation failed: {e}")
            return []


_counterfactual_service = None


def get_counterfactual_service() -> CounterfactualService:
    global _counterfactual_service
    if _counterfactual_service is None:
        _counterfactual_service = CounterfactualService()
    return _counterfactual_service