"""
Tests for src/recourse.py. Needs the trained model, feature_defaults.json,
and data/processed/X_test.csv + X_test_fe_v2.csv.
"""
import json
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

from src.decision import decide
from src.inference import Scorer
from src.recourse import ALLOWED_FEATURES, BAND, DERIVED, apply_changes, find_recourse


@pytest.fixture(scope="module")
def scorer():
    return Scorer()


@pytest.fixture(scope="module")
def sample(scorer):
    base = pd.read_csv("data/processed/X_test.csv").head(600)
    fe = pd.read_csv("data/processed/X_test_fe_v2.csv").head(600)
    probs = scorer.model.predict(fe[scorer.features])
    return base, decide(probs, scorer.approve_below, scorer.reject_from)


@pytest.fixture(scope="module")
def solved(scorer, sample):
    base, dec = sample
    out = []
    for i in np.where(dec == "reject")[0][:15]:
        raw = base.iloc[i].to_dict()
        out.append((raw, find_recourse(scorer, raw)))
    return out


def test_recourse_found_for_some_rejects(solved):
    assert any(len(res) > 0 for _, res in solved)


def test_every_recourse_verified_by_independent_rescore(scorer, solved):
    """Applying the suggested changes and scoring from scratch must give the claimed result."""
    for raw, res in solved:
        for cf in res:
            out = scorer.score(apply_changes(raw, cf["changes"]))
            assert out["probability"] == pytest.approx(cf["new_probability"], abs=1e-6)
            assert out["decision"].upper() == cf["new_decision"]
            assert BAND[out["decision"]] < BAND["reject"]


def test_changes_respect_bounds(solved):
    for _, res in solved:
        for cf in res:
            ch = {c["feature"]: c for c in cf["changes"]}
            if "AMT_CREDIT" in ch:
                assert ch["AMT_CREDIT"]["recommended"] / ch["AMT_CREDIT"]["original"] >= 0.5 - 1e-9
            if "AMT_INCOME_TOTAL" in ch:
                ratio = ch["AMT_INCOME_TOTAL"]["recommended"] / ch["AMT_INCOME_TOTAL"]["original"]
                assert 1.0 <= ratio <= 1.5 + 1e-9
            if "AMT_ANNUITY" in ch:
                ratio = ch["AMT_ANNUITY"]["recommended"] / ch["AMT_ANNUITY"]["original"]
                assert 0.35 - 1e-9 <= ratio <= 1.0 + 1e-9  # 0.5 credit x 0.7 tenure


def test_only_allowed_features_change(solved):
    for raw, res in solved:
        for cf in res:
            assert {c["feature"] for c in cf["changes"]} <= ALLOWED_FEATURES
            mod = apply_changes(raw, cf["changes"])
            for k, v in raw.items():
                if k in ALLOWED_FEATURES or k in DERIVED:
                    continue
                assert mod[k] == v, f"immutable field {k} was altered"


def test_approved_applicant_gets_no_recourse(scorer, sample):
    base, dec = sample
    i = int(np.where(dec == "approve")[0][0])
    assert find_recourse(scorer, base.iloc[i].to_dict()) == []


def test_results_ranked_and_distinct(solved):
    for _, res in solved:
        assert len(res) <= 3
        costs = [cf["cost"] for cf in res]
        assert costs == sorted(costs)
        combos = [tuple(cf["levers"]) for cf in res]
        assert len(combos) == len(set(combos))


def test_results_are_json_serialisable(solved):
    for _, res in solved:
        json.dumps(res)