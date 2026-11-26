"""
Hand-checked values for the evaluation metrics (no model or data needed).
"""
import numpy as np
import pandas as pd
import pytest

from src.explanation_eval import agreement, jaccard
from src.recourse_eval import aggregate


def test_agreement_identical_vectors():
    v = np.array([5.0, -4, 3, -2, 1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01])
    a = agreement(v, v, k_rank=10)
    assert a["top5_overlap"] == 1.0
    assert a["top10_overlap"] == 1.0
    assert a["spearman"] == pytest.approx(1.0)
    assert a["sign_agreement_top5"] == 1.0


def test_agreement_flipped_signs():
    v = np.array([5.0, -4, 3, -2, 1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01])
    a = agreement(-v, v, k_rank=10)
    assert a["top5_overlap"] == 1.0          # same features by magnitude
    assert a["sign_agreement_top5"] == 0.0   # every direction wrong
    assert a["spearman"] == pytest.approx(-1.0)


def test_agreement_partial_overlap():
    shap_v = np.array([10.0, 9, 8, 7, 6, 0, 0, 0, 0, 0, 0, 0])
    lime_w = np.array([10.0, 9, 8, 0, 0, 7, 6, 0, 0, 0, 0, 0])
    assert agreement(lime_w, shap_v)["top5_overlap"] == pytest.approx(3 / 5)


def test_jaccard():
    assert jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(2 / 4)


def test_aggregate_hand_example():
    df = pd.DataFrame({
        "found":     [True, True, False, True],
        "improves":  [True, True, None, True],
        "cost":      [0.1, 0.3, None, 0.2],
        "levers":    [1, 2, None, 1],
        "plausible": [True, False, None, True],
        "robust":    [True, True, None, False],
        "latency_ms": [10.0, 20.0, 30.0, 40.0],
    })
    for c in ("improves", "plausible", "robust"):
        df[c] = df[c].astype("boolean")
    a = aggregate(df)
    assert a["coverage"] == 0.75
    assert a["mean_cost"] == pytest.approx(0.2)
    assert a["median_cost"] == pytest.approx(0.2)
    assert a["mean_levers"] == pytest.approx(4 / 3, abs=1e-3)
    assert a["plausible_share"] == pytest.approx(2 / 3, abs=1e-4)
    assert a["robust_share"] == pytest.approx(2 / 3, abs=1e-4)
    assert a["latency_ms_mean"] == 25.0
