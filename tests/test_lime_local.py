"""
Tests for src/lime_local.py. Needs the trained model and the FE v2 data files.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

REQUIRED = ["models/final_lgbm.pkl", "models/final_model_card.json",
            "data/processed/X_train_fe_v2.csv", "data/processed/X_test_fe_v2.csv"]
if not all(Path(p).exists() for p in REQUIRED):
    pytest.skip("model/data files missing", allow_module_level=True)

import joblib
import shap

from src.lime_local import DEFAULT_K, LocalLime

N = 8


@pytest.fixture(scope="module")
def setup():
    feats = json.load(open("models/final_model_card.json"))["features_used"]
    model = joblib.load("models/final_lgbm.pkl")
    bg = pd.read_csv("data/processed/X_train_fe_v2.csv", nrows=3000)[feats]
    X = pd.read_csv("data/processed/X_test_fe_v2.csv", nrows=N)[feats].astype(float)
    sv = shap.TreeExplainer(model).shap_values(X)
    sv = np.asarray(sv[-1] if isinstance(sv, list) else sv)
    lime = LocalLime(model, feats, bg, num_samples=2000)
    weights = [lime.explain(X.iloc[i].values)[1] for i in range(N)]
    return lime, X, sv, weights


def test_explains_top_gain_subset(setup):
    lime, *_ = setup
    assert len(lime.subset) == DEFAULT_K
    assert len(set(lime.subset)) == DEFAULT_K


def test_same_seed_same_answer(setup):
    lime, X, _, _ = setup
    a = lime.explain(X.iloc[0].values, seed=7)[1]
    b = lime.explain(X.iloc[0].values, seed=7)[1]
    assert np.allclose(a, b)


def test_rows_sorted_and_limited(setup):
    lime, X, _, _ = setup
    rows, _, score = lime.explain(X.iloc[0].values, num_features=6)
    assert len(rows) == 6
    w = [abs(r["weight"]) for r in rows]
    assert w == sorted(w, reverse=True)
    assert {"feature", "rule", "weight"} <= set(rows[0])
    assert np.isfinite(score)


def test_explanations_differ_between_applicants(setup):
    _, _, _, weights = setup
    assert not all(np.allclose(weights[0], w) for w in weights[1:])


def test_signs_match_shap(setup):
    """Guards the label-1 read-out: LIME's label 0 holds negated weights in regression mode."""
    lime, _, sv, weights = setup
    agree = []
    for i in range(N):
        s = sv[i][lime.idx]
        top = np.argsort(-np.abs(s))[:3]
        agree.append(np.mean(np.sign(weights[i][top]) == np.sign(s[top])))
    assert np.mean(agree) >= 0.6


def test_closer_to_own_shap_than_to_another_applicants(setup):
    lime, _, sv, weights = setup
    own = [spearmanr(weights[i], sv[i][lime.idx]).correlation for i in range(N)]
    other = [spearmanr(weights[i], sv[(i + 1) % N][lime.idx]).correlation for i in range(N)]
    assert np.mean(own) > np.mean(other)
