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
from src.recourse import (ALLOWED_FEATURES, BAND, DEFAULT_MARGIN, DERIVED, Plausibility,
                          apply_changes, default_plausibility, explain_change, find_recourse)


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

# --- Phase 4.7: margin, plausibility, refinement, Review -> Approve tier ---

def test_every_option_clears_the_margin(solved):
    for _, res in solved:
        for cf in res:
            assert cf["robust_margin"] >= DEFAULT_MARGIN - 1e-9


def test_options_are_plausible_when_reference_exists(solved):
    if default_plausibility() is None:
        pytest.skip("run python3 -m src.build_recourse_reference")
    for _, res in solved:
        for cf in res:
            assert cf["plausible"] is True


def test_refinement_never_costs_more(scorer, solved):
    for raw, res in solved[:8]:
        coarse = {tuple(cf["levers"]): cf["cost"] for cf in find_recourse(scorer, raw, refine=False)}
        for cf in res:
            key = tuple(cf["levers"])
            if key in coarse:
                assert cf["cost"] <= coarse[key] + 1e-9


@pytest.fixture(scope="module")
def review_solved(scorer, sample):
    base, dec = sample
    out = []
    for i in np.where(dec == "review")[0][:25]:
        raw = base.iloc[i].to_dict()
        out.append((raw, find_recourse(scorer, raw)))
    return out


def test_review_applicants_get_paths_to_approve(scorer, review_solved):
    assert any(res for _, res in review_solved)
    for raw, res in review_solved:
        for cf in res:
            assert cf["new_decision"] == "APPROVE"
            assert scorer.score(apply_changes(raw, cf["changes"]))["decision"] == "approve"


def test_explain_change_adds_up_to_the_score_change(scorer, solved):
    import shap
    raw, res = next((r, x) for r, x in solved if x)
    deltas = explain_change(scorer, shap.TreeExplainer(scorer.model), raw, res[0], top=len(scorer.features),
                            only_changed=False)
    before = scorer.model.predict(scorer.frame(raw), raw_score=True)[0]
    after = scorer.model.predict(scorer.frame(apply_changes(raw, res[0]["changes"])), raw_score=True)[0]
    assert sum(d["delta"] for d in deltas) == pytest.approx(after - before, abs=1e-6)


def test_plausibility_on_toy_reference():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(500, 4))
    p = Plausibility(ref, ref.mean(axis=0), ref.std(axis=0), threshold=0.5)
    near, far = p.distance(np.array([[0.0, 0, 0, 0], [8.0, 8, 8, 8]]))
    assert near <= p.threshold < far


# --- Phase 4.7c: waiting lever and applicant preferences ---

@pytest.fixture(scope="module")
def wait_option(scorer, sample):
    """First rejected or Review applicant whose options include waiting."""
    base, dec = sample
    for i in np.where(dec != "approve")[0][:80]:
        raw = base.iloc[i].to_dict()
        for cf in find_recourse(scorer, raw):
            if cf["wait_months"] > 0:
                return raw, cf
    pytest.skip("no waiting option in the sample")


def test_waiting_option_is_verified_and_spelled_out(scorer, wait_option):
    raw, cf = wait_option
    feats = {c["feature"] for c in cf["changes"]}
    assert "WAIT_MONTHS" in feats
    assert "wait" in cf["levers"]
    assert cf["wait_months"] in (3, 6, 12)
    assert "DAYS_BIRTH" not in feats and "AGE_YEARS" not in feats   # age held fixed
    out = scorer.score(apply_changes(raw, cf["changes"]))
    assert out["probability"] == pytest.approx(cf["new_probability"], abs=1e-6)


def test_locked_levers_are_not_used(scorer, solved):
    for raw, _ in solved[:8]:
        for cf in find_recourse(scorer, raw, locked=("income", "wait")):
            assert "income" not in cf["levers"] and "wait" not in cf["levers"]
        assert find_recourse(scorer, raw, locked=("credit", "income", "tenure", "wait")) == []


def test_loan_reduction_and_wait_caps(scorer, solved):
    for raw, _ in solved[:8]:
        for cf in find_recourse(scorer, raw, max_loan_reduction=0.1, max_wait_months=3):
            assert cf["wait_months"] <= 3
            ch = {c["feature"]: c for c in cf["changes"]}
            if "AMT_CREDIT" in ch:
                assert ch["AMT_CREDIT"]["recommended"] / ch["AMT_CREDIT"]["original"] >= 0.9 - 1e-9


def test_unknown_lever_rejected(scorer, solved):
    with pytest.raises(ValueError):
        find_recourse(scorer, solved[0][0], locked=("savings",))


def test_no_returned_option_is_dominated(solved, review_solved):
    from src.recourse import _dominates

    def scales(cf):
        ch = {c["feature"]: c for c in cf["changes"]}
        c = ch["AMT_CREDIT"]["recommended"] / ch["AMT_CREDIT"]["original"] if "AMT_CREDIT" in ch else 1.0
        i = ch["AMT_INCOME_TOTAL"]["recommended"] / ch["AMT_INCOME_TOTAL"]["original"] if "AMT_INCOME_TOTAL" in ch else 1.0
        a = (ch["AMT_ANNUITY"]["recommended"] / ch["AMT_ANNUITY"]["original"]) / c if "AMT_ANNUITY" in ch else 1.0
        return (round(c, 9), round(i, 9), round(a, 9), cf["wait_months"])

    for _, res in list(solved) + list(review_solved):
        s = [scales(cf) for cf in res]
        for x in s:
            assert not any(_dominates(y, x) for y in s if y is not x)


def test_drivers_only_name_features_that_change(scorer, solved):
    import shap
    explainer = shap.TreeExplainer(scorer.model)
    for raw, res in solved[:5]:
        for cf in res:
            before = scorer.frame(raw).iloc[0]
            after = scorer.frame(apply_changes(raw, cf["changes"])).iloc[0]
            for d in explain_change(scorer, explainer, raw, cf):
                assert not np.isclose(before[d["feature"]], after[d["feature"]], equal_nan=True)
                assert not d["feature"].startswith("AGE")


# --- Phase 5.2b: ensemble-robust search ---

class _ConstantModel:
    def __init__(self, p):
        self.p = p

    def predict(self, X):
        return np.full(len(X), self.p)


def _ensemble(p, n=5):
    from src.robustness import Ensemble
    return Ensemble([_ConstantModel(p) for _ in range(n)], [(0.0349, 0.1189)] * n, (0.0349, 0.1189))


def test_agreeable_ensemble_changes_nothing(scorer, solved):
    """Retrained models that approve everything never block an option."""
    for raw, res in solved[:5]:
        with_ens = find_recourse(scorer, raw, ensemble=_ensemble(0.0))
        assert [o["changes"] for o in with_ens] == [o["changes"] for o in res]
        assert all(o["ensemble_agreement"] == "5/5" for o in with_ens)


def test_disagreeing_ensemble_blocks_everything(scorer, solved):
    """Retrained models that reject everything leave no option."""
    for raw, _ in solved[:5]:
        assert find_recourse(scorer, raw, ensemble=_ensemble(0.9)) == []


def test_min_agree_validated(scorer, solved):
    with pytest.raises(ValueError):
        find_recourse(scorer, solved[0][0], ensemble=_ensemble(0.0), min_agree=6)
