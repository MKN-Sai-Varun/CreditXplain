"""
Group metrics and per-group cutoffs of src/age_fairness_experiment.py, on hand-made data.
"""
import numpy as np
import pytest

from src.age_fairness_experiment import age_bands, decide_per_group, gaps, group_table, per_group_cutoffs


def test_age_bands_match_the_audit():
    assert list(age_bands([20, 25, 26, 35, 50, 55, 56])) == ["<25", "<25", "25-35", "25-35", "45-55", "45-55", "55+"]


def test_group_table_hand_example():
    y = [0, 0, 1, 0, 1, 1]
    probs = [0.01, 0.2, 0.3, 0.01, 0.02, 0.4]
    bands = np.array(["approve", "review", "reject", "approve", "approve", "reject"])
    groups = ["<25", "<25", "<25", "55+", "55+", "55+"]
    rows = {r["group"]: r for r in group_table(y, probs, bands, groups)}
    young, old = rows["<25"], rows["55+"]
    assert young["n"] == 3 and young["actual_default_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert young["approve_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert young["approve_rate_among_repaid"] == 0.5             # 1 of the 2 who repaid
    assert young["default_rate_among_approved"] == 0.0
    assert old["approve_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert old["approve_rate_among_repaid"] == 1.0
    assert old["default_rate_among_approved"] == 0.5
    assert young["auc"] == 1.0
    assert young["mean_predicted_risk"] == pytest.approx(0.17, abs=1e-4)
    g = gaps(list(rows.values()))
    assert g["demographic_parity_difference"] == pytest.approx(1 / 3, abs=1e-3)
    assert g["equal_opportunity_difference"] == 0.5
    assert g["actual_default_rate_spread"] == pytest.approx(1 / 3, abs=1e-3)


def test_per_group_cutoffs_equalise_rates():
    rng = np.random.default_rng(0)
    probs = np.concatenate([rng.uniform(0, 0.1, 1000), rng.uniform(0.05, 0.3, 1000)])   # second group riskier
    groups = np.array(["a"] * 1000 + ["b"] * 1000)
    cut = per_group_cutoffs(probs, groups, approve_rate=0.3, reject_rate=0.2)
    bands = decide_per_group(probs, groups, cut)
    for g in ("a", "b"):
        m = groups == g
        assert (bands[m] == "approve").mean() == pytest.approx(0.3, abs=0.01)
        assert (bands[m] == "reject").mean() == pytest.approx(0.2, abs=0.01)
    assert cut["b"][0] > cut["a"][0]                               # riskier group gets a higher cutoff
