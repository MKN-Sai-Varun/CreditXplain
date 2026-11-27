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


def test_aggregate_robustness_fields():
    df = pd.DataFrame({
        "found":     [True, True, True, False],
        "improves":  [True, True, True, None],
        "cost":      [0.1, 0.2, 0.3, None],
        "levers":    [1, 1, 2, None],
        "plausible": [True, True, False, None],
        "robust":    [True, True, True, None],
        "latency_ms": [1.0, 1.0, 1.0, 1.0],
        "n_models":  [5, 5, 5, 5],
        "holds_own": [5, 3, 5, None],
        "holds_fixed": [5, 5, 4, None],
        "base_holds_own": [0, 1, 0, 0],
    })
    for c in ("improves", "plausible", "robust"):
        df[c] = df[c].astype("boolean")
    a = aggregate(df)
    assert a["retrained_models"] == 5
    assert a["holds_on_all_retrained"] == pytest.approx(2 / 3, abs=1e-4)
    assert a["holds_on_all_retrained_fixed_cutoffs"] == pytest.approx(2 / 3, abs=1e-4)
    assert a["mean_retrained_models_held"] == pytest.approx(13 / 3, abs=1e-3)
    assert a["usable_and_holds_on_all"] == pytest.approx(1 / 4)       # only the first row
    assert a["unchanged_applicant_already_promised_on_any"] == pytest.approx(1 / 4)


def test_latency_summary():
    from src.system_latency import summarize
    s = summarize([10.0, 20.0, 30.0, None, 40.0])
    assert s["n"] == 4
    assert s["mean_ms"] == 25.0 and s["median_ms"] == 25.0 and s["max_ms"] == 40.0
    assert 30.0 <= s["p95_ms"] <= 40.0
    assert summarize([None, None]) is None


def test_time_cost_comparison():
    from src.time_cost_sensitivity import compare

    def opt(levers, wait, credit):
        return {"levers": levers, "wait_months": wait,
                "changes": [{"feature": "AMT_CREDIT", "recommended": credit}]}

    a, b, c = opt(["credit"], 0, 100.0), opt(["credit", "wait"], 6, 120.0), opt(["income"], 0, 0.0)
    runs = {
        0.04: [[a, b], [c], []],           # baseline
        0.02: [[b, a], [c], []],           # waiting now ranked first for applicant 1
        0.08: [[a, b], [c], []],           # identical to baseline
    }
    out = compare(runs, 0.04)
    assert out["0.04"]["same_first_option_as_baseline"] == 1.0
    assert out["0.08"]["same_options_shown_as_baseline"] == 1.0
    assert out["0.02"]["same_first_option_as_baseline"] == 0.5      # 1 of the 2 applicants with options
    assert out["0.02"]["same_options_shown_as_baseline"] == 1.0     # same set, different order
    assert out["0.02"]["first_option_waits"] == 0.5
    assert out["0.04"]["coverage"] == round(2 / 3, 4)
