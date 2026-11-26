"""
Counterfactual Service - bounded recourse search over applicant-controlled levers
"""
from typing import Any, Dict, List

from backend.services.explainability_service import get_explainability_service
from backend.services.prediction_service import get_prediction_service
from src.recourse import explain_change, find_recourse


class CounterfactualService:
    def __init__(self):
        self.ps = get_prediction_service()

    def generate_counterfactuals(self, features: Dict[str, Any], num_counterfactuals: int = 3) -> List[Dict[str, Any]]:
        if self.ps.scorer is None:
            return []
        try:
            options = find_recourse(self.ps.scorer, features, n=num_counterfactuals)
            if options:
                shap_explainer = get_explainability_service().shap_explainer
                if shap_explainer is not None:
                    for opt in options:
                        opt["drivers"] = explain_change(self.ps.scorer, shap_explainer, features, opt)
            return options
        except Exception as e:
            print(f"⚠ Counterfactual generation failed: {e}")
            return []


_counterfactual_service = None


def get_counterfactual_service() -> CounterfactualService:
    global _counterfactual_service
    if _counterfactual_service is None:
        _counterfactual_service = CounterfactualService()
    return _counterfactual_service