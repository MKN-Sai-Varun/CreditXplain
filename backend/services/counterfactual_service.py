"""
Counterfactual Service - bounded recourse search over applicant-controlled levers.

Options that survive retrained models (bootstrap models 1-5) are listed first,
marked retrain_check_passed; standard options fill the remaining slots. Without
those model files, or with ROBUST_RECOURSE off, it is the standard search.
"""
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.services.explainability_service import get_explainability_service
from backend.services.prediction_service import get_prediction_service
from src.recourse import explain_change, find_recourse_hybrid
from src.robustness import Ensemble

SEARCH_MODELS = range(1, 6)


class CounterfactualService:
    def __init__(self):
        self.ps = get_prediction_service()
        self.ensemble = None
        if settings.ROBUST_RECOURSE:
            try:
                self.ensemble = Ensemble.load(ks=SEARCH_MODELS)
            except Exception as e:
                print(f"⚠ Retrained models not loaded, robust recourse off: {e}")
            if self.ensemble is not None:
                print(f"✓ Robust recourse on ({len(self.ensemble)} retrained models)")

    def generate_counterfactuals(self, features: Dict[str, Any], num_counterfactuals: int = 3,
                                 preferences: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """preferences: keyword arguments for find_recourse (locked, max_loan_reduction, max_wait_months)."""
        if self.ps.scorer is None:
            return []
        try:
            options = find_recourse_hybrid(self.ps.scorer, features, self.ensemble,
                                           n=num_counterfactuals, **(preferences or {}))
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