"""
Inference pipeline: raw applicant fields -> FE v2 features -> final model.
"""
import json
import re
from pathlib import Path

import joblib
import pandas as pd

from src.feature_engineering_v2 import add_features
from src.decision import decide_one

ROOT = Path(__file__).resolve().parents[1]


class Scorer:
    def __init__(self, model_path=None, card_path=None, defaults_path=None):
        model_path = model_path or ROOT / "models" / "final_lgbm.pkl"
        card_path = card_path or ROOT / "models" / "final_model_card.json"
        defaults_path = defaults_path or ROOT / "models" / "feature_defaults.json"

        self.model = joblib.load(model_path)
        with open(card_path) as f:
            card = json.load(f)
        with open(defaults_path) as f:
            self.defaults = json.load(f)

        self.features = card["features_used"]
        self.approve_below = card["approve_threshold"]
        self.reject_from = card["reject_threshold"]

    def build_row(self, raw):
        row = dict(self.defaults)

        for key, val in raw.items():
            if val is None:
                continue
            if isinstance(val, str):
                # categorical: clear the one-hot group, then set the chosen level
                prefix = key + "_"
                for col in row:
                    if col.startswith(prefix):
                        row[col] = 0.0
                chosen = prefix + re.sub(r"[^A-Za-z0-9_]+", "_", val)
                if chosen in row:
                    row[chosen] = 1.0
            elif key in row:
                row[key] = float(val)

        # derived base columns, only when the caller did not supply them
        if "AGE_YEARS" not in raw and raw.get("DAYS_BIRTH") is not None:
            row["AGE_YEARS"] = float(int(-float(raw["DAYS_BIRTH"]) / 365))
        if "CREDIT_TO_GOODS_RATIO" not in raw and row["AMT_GOODS_PRICE"] > 0:
            row["CREDIT_TO_GOODS_RATIO"] = row["AMT_CREDIT"] / row["AMT_GOODS_PRICE"]
        if "CREDIT_INCOME_RATIO" not in raw and row["AMT_INCOME_TOTAL"] > 0:
            row["CREDIT_INCOME_RATIO"] = row["AMT_CREDIT"] / row["AMT_INCOME_TOTAL"]

        return row

    def frame(self, raw):
        df = add_features(pd.DataFrame([self.build_row(raw)]))
        return df.reindex(columns=self.features).astype(float)

    def score(self, raw):
        X = self.frame(raw)
        prob = float(self.model.predict(X)[0])
        return {
            "probability": prob,
            "decision": decide_one(prob, self.approve_below, self.reject_from),
        }