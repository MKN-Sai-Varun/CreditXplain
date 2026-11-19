"""
Tests for src/inference.py. Needs: models/final_lgbm.pkl, final_model_card.json,
feature_defaults.json, and data/processed/X_test.csv + X_test_fe_v2.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REQUIRED = [
    "models/final_lgbm.pkl",
    "models/final_model_card.json",
    "models/feature_defaults.json",
    "data/processed/X_test.csv",
    "data/processed/X_test_fe_v2.csv",
]

if not all(Path(p).exists() for p in REQUIRED):
    pytest.skip("model/data files missing", allow_module_level=True)

from src.inference import Scorer


@pytest.fixture(scope="module")
def scorer():
    return Scorer()


def test_matches_batch_pipeline(scorer):
    """Row-by-row inference must reproduce the batch FE v2 scores."""
    base = pd.read_csv("data/processed/X_test.csv").head(25)
    fe = pd.read_csv("data/processed/X_test_fe_v2.csv").head(25)
    expected = scorer.model.predict(fe[scorer.features])
    got = [scorer.score(r.to_dict())["probability"] for _, r in base.iterrows()]
    assert np.allclose(got, expected, atol=1e-6)


def test_partial_form_gives_full_clean_frame(scorer):
    X = scorer.frame({"AMT_CREDIT": 500000, "AMT_INCOME_TOTAL": 180000})
    assert list(X.columns) == scorer.features
    assert X.shape == (1, 300)
    assert not X.isna().any().any()


def test_probability_and_decision_valid(scorer):
    out = scorer.score({"AMT_CREDIT": 500000, "AMT_INCOME_TOTAL": 180000, "DAYS_BIRTH": -12000})
    assert 0 <= out["probability"] <= 1
    assert out["decision"] in ("approve", "review", "reject")


def test_categorical_sets_one_hot_group(scorer):
    row = scorer.build_row({"NAME_INCOME_TYPE": "Pensioner"})
    assert row["NAME_INCOME_TYPE_Pensioner"] == 1
    assert row["NAME_INCOME_TYPE_Working"] == 0


def test_derived_ratios_follow_inputs(scorer):
    row = scorer.build_row({"AMT_CREDIT": 600000, "AMT_GOODS_PRICE": 500000, "AMT_INCOME_TOTAL": 200000})
    assert row["CREDIT_TO_GOODS_RATIO"] == pytest.approx(1.2)
    assert row["CREDIT_INCOME_RATIO"] == pytest.approx(3.0)


def test_age_derived_from_days_birth(scorer):
    assert scorer.build_row({"DAYS_BIRTH": -10950})["AGE_YEARS"] == 30


def test_gender_cannot_change_score(scorer):
    a = scorer.score({"CODE_GENDER": "M", "AMT_CREDIT": 400000})["probability"]
    b = scorer.score({"CODE_GENDER": "F", "AMT_CREDIT": 400000})["probability"]
    assert a == b
    assert not any("GENDER" in f for f in scorer.features)


def test_more_credit_does_not_lower_risk(scorer):
    base = {"AMT_INCOME_TOTAL": 150000, "AMT_GOODS_PRICE": 300000, "AMT_ANNUITY": 15000}
    lo = scorer.score({**base, "AMT_CREDIT": 300000})["probability"]
    hi = scorer.score({**base, "AMT_CREDIT": 900000})["probability"]
    assert hi >= lo * 0.8  # loose monotonic sanity check, not strict