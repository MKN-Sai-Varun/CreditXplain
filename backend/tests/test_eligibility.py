"""
Eligibility rules before the model: minors apply with a guardian, applicants
with no income are never approved automatically.
"""
import json
from pathlib import Path

import pytest

from backend.services import eligibility_service as el

ADULT = {"AMT_CREDIT": 200000, "DAYS_BIRTH": -round(35 * 365.25), "AMT_INCOME_TOTAL": 180000,
         "NAME_INCOME_TYPE": "Working", "DAYS_EMPLOYED": -1800}
MINOR = {**ADULT, "DAYS_BIRTH": -round(16 * 365.25), "AMT_INCOME_TOTAL": 0, "NAME_INCOME_TYPE": "Student",
         "DAYS_EMPLOYED": 0}
GUARDIAN = {"full_name": "Parent", "relationship": "parent", "age": 45, "yearly_income": 300000,
            "income_type": "Working", "years_employed": 10}
PDF = b"%PDF-1.4\n%test document\n"


# ---------------------------------------------------------------- the rules

def test_adult_with_income_has_no_rules():
    e = el.check(ADULT)
    assert e["rules"] == [] and e["automatic_approval"] and e["assessed_on"] == "applicant"
    assert el.assessed_features(ADULT, None) == ADULT


def test_minor_is_assessed_on_the_guardian():
    e = el.check(MINOR, GUARDIAN)
    assert [r["code"] for r in e["rules"]] == ["minor"]       # guardian has an income
    assert e["assessed_on"] == "guardian" and not e["automatic_approval"]
    assert e["required_documents"] == ["guardian_id_proof", "guardian_income_proof"]
    f = el.assessed_features(MINOR, GUARDIAN)
    assert f["AMT_INCOME_TOTAL"] == 300000 and f["NAME_INCOME_TYPE"] == "Working"
    assert f["DAYS_BIRTH"] == -round(45 * 365.25) and f["DAYS_EMPLOYED"] == -round(10 * 365.25)
    assert f["AMT_CREDIT"] == MINOR["AMT_CREDIT"]


def test_guardian_replaces_one_hot_and_derived_columns():
    raw = {**MINOR, "AGE_YEARS": 16.0, "CREDIT_INCOME_RATIO": 99.0, "NAME_INCOME_TYPE_Student": 1}
    del raw["NAME_INCOME_TYPE"]
    f = el.assessed_features(raw, GUARDIAN)
    assert "AGE_YEARS" not in f and "CREDIT_INCOME_RATIO" not in f and "NAME_INCOME_TYPE_Student" not in f
    assert el.income_type(raw) == "Student"


def test_minor_with_guardian_without_income_gets_both_rules():
    e = el.check(MINOR, {**GUARDIAN, "yearly_income": 0, "income_type": "Unemployed"})
    assert [r["code"] for r in e["rules"]] == ["minor", "no_income"]


@pytest.mark.parametrize("change", [{"AMT_INCOME_TOTAL": 0}, {"NAME_INCOME_TYPE": "Student"},
                                    {"NAME_INCOME_TYPE": "Unemployed"}, {"AMT_INCOME_TOTAL": None}])
def test_no_income_removes_automatic_approval(change):
    e = el.check({**ADULT, **change})
    assert [r["code"] for r in e["rules"]] == ["no_income"]
    assert e["assessed_on"] == "applicant"


def test_rules_only_remove_automatic_approval():
    blocked = el.check({**ADULT, "NAME_INCOME_TYPE": "Student"})
    assert el.route("APPROVE", blocked) == "REVIEW"
    assert el.route("REVIEW", blocked) == "REVIEW"
    assert el.route("REJECT", blocked) == "REJECT"
    assert el.route("APPROVE", el.check(ADULT)) == "APPROVE"
    assert el.recourse_applies("REJECT", blocked)
    assert not el.recourse_applies("REVIEW", blocked)          # automatic approval is out of reach
    assert el.recourse_applies("REVIEW", el.check(ADULT))


def test_age_boundary_and_missing_age():
    assert not el.is_minor({"DAYS_BIRTH": -round(18 * 365.25)})
    assert not el.is_minor({"DAYS_BIRTH": -6574})              # what the form sends for age 18
    assert el.is_minor({"DAYS_BIRTH": -round(17.9 * 365.25)})
    assert not el.is_minor({}) and el.applicant_age({"DAYS_BIRTH": 5}) is None


# ---------------------------------------------------------------- through the API

def test_minor_needs_guardian(client, applicant_headers):
    r = client.post("/api/applications/", json={"features": MINOR}, headers=applicant_headers)
    assert r.status_code == 422 and "guardian" in r.text

    r = client.post("/api/applications/", json={"features": MINOR, "guardian": GUARDIAN}, headers=applicant_headers)
    assert r.status_code == 201, r.text
    detail = client.get(f"/api/applications/{r.json()['id']}", headers=applicant_headers).json()
    assert detail["guardian"]["full_name"] == "Parent"


@pytest.mark.parametrize("bad", [{"age": 16}, {"relationship": "friend"}, {"income_type": "Lottery"},
                                 {"yearly_income": -1}])
def test_guardian_validation(client, applicant_headers, bad):
    r = client.post("/api/applications/", json={"features": MINOR, "guardian": {**GUARDIAN, **bad}},
                    headers=applicant_headers)
    assert r.status_code == 422


def test_guardian_ignored_for_adults(client, applicant_headers):
    r = client.post("/api/applications/", json={"features": ADULT, "guardian": GUARDIAN}, headers=applicant_headers)
    assert r.status_code == 201
    assert client.get(f"/api/applications/{r.json()['id']}", headers=applicant_headers).json()["guardian"] is None


def test_officer_needs_guardian_documents_to_approve_a_minor(client, applicant_headers, officer_headers,
                                                             scored_application):
    app_id = scored_application(features=MINOR)
    act = lambda: client.post(f"/api/officer/{app_id}/action",
                              json={"action": "approve", "justification": "Guardian income verified"},
                              headers=officer_headers)
    r = act()
    assert r.status_code == 400 and "guardian_id_proof" in r.text and "guardian_income_proof" in r.text

    for doc_type in el.GUARDIAN_DOCS:
        up = client.post(f"/api/applications/{app_id}/documents", files={"file": ("doc.pdf", PDF, "application/pdf")},
                         data={"doc_type": doc_type}, headers=applicant_headers)
        assert up.status_code == 201, up.text
    assert act().status_code == 200


DATA = ["models/final_lgbm.pkl", "data/processed/X_test.csv", "data/processed/X_test_fe_v2.csv"]


@pytest.mark.skipif(not all(Path(p).exists() for p in DATA), reason="model/data missing")
def test_real_decision_for_minor_and_no_income(client, applicant_headers, officer_headers):
    import numpy as np
    import pandas as pd
    from backend.services.prediction_service import get_prediction_service
    from src.decision import decide

    sc = get_prediction_service().scorer
    base = pd.read_csv("data/processed/X_test.csv").head(400)
    fe = pd.read_csv("data/processed/X_test_fe_v2.csv").head(400)
    bands = decide(sc.model.predict(fe[sc.features]), sc.approve_below, sc.reject_from)
    approved = json.loads(base.iloc[int(np.where(bands == "approve")[0][0])].to_json())

    # a model-approved adult marked as a student is sent to an officer, without Review -> Approve options
    student = {**{k: v for k, v in approved.items() if not k.startswith("NAME_INCOME_TYPE_")},
               "NAME_INCOME_TYPE": "Student"}
    app_id = client.post("/api/applications/", json={"features": student}, headers=applicant_headers).json()["id"]
    d = client.post(f"/api/predictions/{app_id}", headers=applicant_headers).json()
    assert d["decision"] in ("REVIEW", "REJECT")
    assert [r["code"] for r in d["eligibility"]["rules"]] == ["no_income"]
    if d["decision"] == "REVIEW":
        detail = client.get(f"/api/applications/{app_id}", headers=applicant_headers).json()
        assert detail["explanation"]["counterfactuals"] is None
        assert client.post(f"/api/predictions/{app_id}/recourse", json={},
                           headers=applicant_headers).status_code == 400

    # a minor applying with a guardian is scored on the guardian and never approved automatically
    minor = {**approved, "DAYS_BIRTH": -round(16 * 365.25)}
    app_id = client.post("/api/applications/", json={"features": minor, "guardian": GUARDIAN},
                         headers=applicant_headers).json()["id"]
    d = client.post(f"/api/predictions/{app_id}", headers=applicant_headers).json()
    assert d["decision"] != "APPROVE"
    assert d["eligibility"]["assessed_on"] == "guardian"
    expected = sc.score(el.assessed_features(minor, GUARDIAN))["probability"]
    assert d["probability"] == pytest.approx(expected)

    pdf = client.get(f"/api/applications/{app_id}/report.pdf", headers=applicant_headers).content
    assert b"Eligibility" in pdf and b"Guardian" in pdf
    comp = client.get(f"/api/officer/{app_id}/compliance-report.pdf", headers=officer_headers).content
    assert b"before eligibility rules" in comp
