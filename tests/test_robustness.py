"""
Counting whether recourse holds on retrained models (src/robustness.py). No data needed.
"""
import json

import numpy as np
import pytest

from src.recourse import BAND
from src.robustness import Ensemble, band_index, count_holds


class FixedModel:
    """Stands in for a LightGBM model that always predicts the same probability."""

    def __init__(self, p):
        self.p = p

    def predict(self, X):
        return np.full(1 if X is None else len(X), self.p)


def test_band_index_uses_both_cutoffs():
    assert band_index(0.01, 0.03, 0.12) == BAND["approve"]
    assert band_index(0.03, 0.03, 0.12) == BAND["review"]     # approve is strictly below
    assert band_index(0.12, 0.03, 0.12) == BAND["reject"]     # reject from the cutoff
    assert band_index(0.08, 0.03, 0.12) == BAND["review"]


def test_count_holds_against_promised_band():
    cutoffs = [(0.03, 0.12)] * 4
    probs = [0.01, 0.05, 0.11, 0.2]          # approve, review, review, reject
    assert count_holds(probs, cutoffs, BAND["review"]) == 3     # review or better
    assert count_holds(probs, cutoffs, BAND["approve"]) == 1


def test_ensemble_own_versus_fixed_cutoffs():
    ens = Ensemble([FixedModel(0.11), FixedModel(0.13)],
                   own_cutoffs=[(0.03, 0.10), (0.03, 0.14)],
                   fixed_cutoffs=(0.03, 0.12))
    own, fixed = ens.holds(None, BAND["review"])
    assert own == 1      # 0.11 is rejected under its own 0.10 cutoff; 0.13 is review under 0.14
    assert fixed == 1    # under 0.12: 0.11 is review, 0.13 is rejected
    assert len(ens) == 2


def test_ensemble_load_missing_or_empty(tmp_path):
    assert Ensemble.load(tmp_path / "none.json") is None
    report = tmp_path / "r.json"
    report.write_text(json.dumps({"models": [{"k": 1, "path": str(tmp_path / "gone.pkl"),
                                              "approve_threshold": 0.03, "reject_threshold": 0.12}],
                                  "final_model": {"approve_threshold": 0.03, "reject_threshold": 0.12}}))
    assert Ensemble.load(report) is None     # listed model file is missing


def test_select_cutoffs_rule():
    from src.decision import select_cutoffs
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 0.4, 10000)
    y = (rng.uniform(size=10000) < probs).astype(int)      # default rate rises with probability
    approve_below, reject_from = select_cutoffs(probs, y)
    assert approve_below < reject_from
    assert y[probs < approve_below].mean() <= 0.03 or approve_below == pytest.approx(np.percentile(probs, 40))
    assert reject_from in (pytest.approx(np.percentile(probs, 80)), pytest.approx(np.percentile(probs, 85)))


def test_ensemble_band_matrix():
    ens = Ensemble([FixedModel(0.02), FixedModel(0.08), FixedModel(0.2)],
                   own_cutoffs=[(0.03, 0.12)] * 3, fixed_cutoffs=(0.03, 0.12))
    bands = ens.bands(np.zeros((4, 2)))           # 4 candidate rows
    assert bands.shape == (3, 4)
    assert (bands[0] == BAND["approve"]).all()
    assert (bands[1] == BAND["review"]).all()
    assert (bands[2] == BAND["reject"]).all()


def test_ensemble_load_selects_models_by_number(tmp_path):
    import joblib
    models = []
    for k in range(1, 5):
        path = tmp_path / f"m{k}.pkl"
        joblib.dump(FixedModel(0.01 * k), path)
        models.append({"k": k, "path": str(path), "approve_threshold": 0.03, "reject_threshold": 0.12})
    report = tmp_path / "r.json"
    report.write_text(json.dumps({"models": models,
                                  "final_model": {"approve_threshold": 0.03, "reject_threshold": 0.12}}))
    assert len(Ensemble.load(report)) == 4
    search = Ensemble.load(report, ks=range(1, 3))
    held_out = Ensemble.load(report, ks=range(3, 5))
    assert len(search) == 2 and len(held_out) == 2
    assert [m.p for m in held_out.models] == [0.03, 0.04]
    assert Ensemble.load(report, ks=range(6, 11)) is None


def test_qualification_needs_enough_agreeing_models():
    from src.recourse import _agreeing, _qualifies
    base = BAND["reject"]
    robust = np.array([BAND["review"]] * 3)           # final model: all three candidates qualify
    dist = np.zeros(3)
    ens = np.array([[1, 2, 1],                        # model 1: candidates 0 and 2 reach review
                    [1, 2, 2],                        # model 2: candidate 0 only
                    [0, 1, 2]])                       # model 3: candidates 0 and 1
    assert list(_agreeing(ens, base)) == [3, 1, 1]
    assert list(_qualifies(robust, dist, base, None, ens, 3)) == [True, False, False]
    assert list(_qualifies(robust, dist, base, None, ens, 1)) == [True, True, True]
    assert list(_qualifies(robust, dist, base, None)) == [True, True, True]   # no ensemble: unchanged
