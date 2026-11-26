"""
The waiting lever's record update (src/recourse._pass_time). No model or data needed.
"""
import math

import pytest

from src.recourse import DAYS_PER_MONTH, _pass_time, _search_space

BASE = {
    "DAYS_BIRTH": -12000.0,
    "AGE_YEARS": 32.0,
    "DAYS_EMPLOYED": -1000.0,
    "IS_EMPLOYED_ANOMALY": 0,
    "DAYS_REGISTRATION": -3000.0,
    "DAYS_ID_PUBLISH": -2000.0,
    "DAYS_LAST_PHONE_CHANGE": -500.0,
    "FLAG_OWN_CAR_Y": 1.0,
    "OWN_CAR_AGE": 5.0,
    "AMT_REQ_CREDIT_BUREAU_HOUR": 0.0,
    "AMT_REQ_CREDIT_BUREAU_DAY": 1.0,
    "AMT_REQ_CREDIT_BUREAU_WEEK": 0.0,
    "AMT_REQ_CREDIT_BUREAU_MON": 1.0,
    "AMT_REQ_CREDIT_BUREAU_QRT": 2.0,
    "AMT_REQ_CREDIT_BUREAU_YEAR": 3.0,
}


def test_no_wait_changes_nothing():
    r = _pass_time(BASE, 0)
    assert r == BASE
    assert r is not BASE   # a copy, so callers can modify it safely


def test_days_fields_grow_but_age_does_not():
    r = _pass_time(BASE, 6)
    d = 6 * DAYS_PER_MONTH
    for col in ("DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"):
        assert r[col] == pytest.approx(BASE[col] - d)
    assert r["DAYS_BIRTH"] == BASE["DAYS_BIRTH"]
    assert r["AGE_YEARS"] == BASE["AGE_YEARS"]
    assert r["OWN_CAR_AGE"] == pytest.approx(5.5)


def test_recent_enquiries_move_to_year_window():
    r = _pass_time(BASE, 3)
    for col in ("HOUR", "DAY", "WEEK", "MON", "QRT"):
        assert r["AMT_REQ_CREDIT_BUREAU_" + col] == 0.0
    assert r["AMT_REQ_CREDIT_BUREAU_YEAR"] == 3.0 + 4.0   # 1 + 1 + 2 recent moved in


def test_all_enquiries_gone_after_a_year():
    r = _pass_time(BASE, 12)
    assert all(r[c] == 0.0 for c in r if c.startswith("AMT_REQ_CREDIT_BUREAU_"))


def test_no_tenure_gain_when_not_employed():
    r = _pass_time({**BASE, "IS_EMPLOYED_ANOMALY": 1}, 12)
    assert r["DAYS_EMPLOYED"] == BASE["DAYS_EMPLOYED"]


def test_car_age_unchanged_without_a_car_and_missing_values_kept():
    r = _pass_time({**BASE, "FLAG_OWN_CAR_Y": 0.0, "DAYS_ID_PUBLISH": float("nan")}, 6)
    assert r["OWN_CAR_AGE"] == BASE["OWN_CAR_AGE"]
    assert math.isnan(r["DAYS_ID_PUBLISH"])


def test_original_record_not_modified():
    before = dict(BASE)
    _pass_time(BASE, 12)
    assert BASE == before


def test_search_space_respects_preferences():
    full = _search_space((), 0.5, 12)
    assert full[0] == (1.0, 1.0, 1.0, 0)          # unchanged applicant first
    assert len(full) == 8 * 5 * 4 * 4

    no_wait = _search_space(("wait",), 0.5, 12)
    assert {s[3] for s in no_wait} == {0}

    capped = _search_space((), 0.1, 3)
    assert min(s[0] for s in capped) >= 0.9 - 1e-9
    assert max(s[3] for s in capped) == 3

    nothing = _search_space(("credit", "income", "tenure", "wait"), 0.5, 12)
    assert nothing == [(1.0, 1.0, 1.0, 0)]


@pytest.mark.parametrize("bad", [{"locked": ("savings",)}, {"max_loan_reduction": 0.8},
                                 {"max_loan_reduction": -0.1}])
def test_search_space_rejects_bad_preferences(bad):
    args = {"locked": (), "max_loan_reduction": 0.5, "max_wait_months": 12, **bad}
    with pytest.raises(ValueError):
        _search_space(args["locked"], args["max_loan_reduction"], args["max_wait_months"])
