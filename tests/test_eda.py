"""
Summary helpers of src/eda.py on a hand-made table. No data needed.
"""
import numpy as np
import pandas as pd
import pytest

from src.eda import age_band, default_rate_by, missing_table, top_correlations

DF = pd.DataFrame({
    "SK_ID_CURR": [1, 2, 3, 4, 5, 6],
    "TARGET": [0, 0, 1, 0, 1, 1],
    "SCORE": [0.9, 0.8, 0.2, 0.7, 0.1, 0.3],     # falls as default rises
    "NOISE": [1.0, np.nan, 3.0, np.nan, np.nan, 1.0],
    "GROUP": ["a", "a", "a", "b", "b", "b"],
})


def test_missing_table():
    m = missing_table(DF)
    assert list(m.index) == ["NOISE"]
    assert m.loc["NOISE", "missing"] == 3 and m.loc["NOISE", "share"] == 0.5


def test_age_band_uses_audit_bands():
    days = [-round(a * 365.25) for a in (20, 30, 40, 50, 60)]
    assert list(age_band(days)) == ["<25", "25-35", "35-45", "45-55", "55+"]


def test_default_rate_by_group_and_minimum_size():
    t = default_rate_by(DF, "GROUP")
    assert t.loc["a", "n"] == 3 and t.loc["a", "default_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert t.loc["b", "default_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert default_rate_by(DF, "GROUP", min_count=4).empty


def test_top_correlations_ranks_by_strength_and_skips_id():
    c = top_correlations(DF, k=5)
    assert c.index[0] == "SCORE" and c.loc["SCORE", "spearman"] < -0.8
    assert "SK_ID_CURR" not in c.index and "TARGET" not in c.index
