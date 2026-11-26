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
        return np.array([self.p])


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
