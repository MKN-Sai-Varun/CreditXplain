"""
Explainability Service - SHAP (primary) and LIME (secondary)
"""
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import shap
from backend.services.prediction_service import get_prediction_service
from src.lime_local import LocalLime

TRAIN_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "X_train_fe_v2.csv"


class ExplainabilityService:
    def __init__(self):
        self.ps = get_prediction_service()
        self.shap_explainer = None
        self.lime_explainer = None
        if self.ps.model is None:
            print("⚠ Cannot initialise explainers without a model")
            return
        self.shap_explainer = shap.TreeExplainer(self.ps.model)
        self._init_lime()

    def _init_lime(self):
        if not TRAIN_PATH.exists():
            print("⚠ Training data not found, LIME disabled")
            return
        feats = self.ps.get_feature_names()
        background = pd.read_csv(TRAIN_PATH, nrows=5000)[feats].astype(float)
        self.lime_explainer = LocalLime(self.ps.model, feats, background)

    def generate_shap_explanation(self, features: Dict[str, Any], top_n: int = 20) -> List[Dict[str, Any]]:
        if self.shap_explainer is None:
            return []
        X = self.ps.build_frame(features)
        sv = self.shap_explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv)[0]
        rows = [
            {"feature": f, "value": float(s), "feature_value": float(X.iloc[0][f])}
            for f, s in zip(X.columns, sv)
        ]
        rows.sort(key=lambda r: abs(r["value"]), reverse=True)
        return rows[:top_n]

    def generate_lime_explanation(self, features: Dict[str, Any], num_features: int = 10) -> List[Dict[str, Any]]:
        if self.lime_explainer is None:
            return []
        try:
            X = self.ps.build_frame(features)
            rows, _, _ = self.lime_explainer.explain(X.iloc[0].values, num_features=num_features)
            return rows
        except Exception as e:
            print(f"⚠ LIME failed: {e}")
            return []


_explainability_service = None


def get_explainability_service() -> ExplainabilityService:
    global _explainability_service
    if _explainability_service is None:
        _explainability_service = ExplainabilityService()
    return _explainability_service