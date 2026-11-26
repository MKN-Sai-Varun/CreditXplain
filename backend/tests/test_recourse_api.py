import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REQUIRED = ["models/final_lgbm.pkl", "models/final_model_card.json",
            "models/feature_defaults.json", "data/processed/X_test.csv",
            "data/processed/X_test_fe_v2.csv"]

pytestmark = pytest.mark.skipif(
    not all(Path(p).exists() for p in REQUIRED), reason="model/data missing"
)


def test_rejected_application_returns_valid_recourse(client, applicant_headers):
    from backend.services.prediction_service import get_prediction_service
    from src.decision import decide
    from src.recourse import find_recourse

    sc = get_prediction_service().scorer
    base = pd.read_csv("data/processed/X_test.csv").head(600)
    fe = pd.read_csv("data/processed/X_test_fe_v2.csv").head(600)
    dec = decide(sc.model.predict(fe[sc.features]), sc.approve_below, sc.reject_from)

    chosen = None
    for i in np.where(dec == "reject")[0][:40]:
        raw = json.loads(base.iloc[i].to_json())
        if find_recourse(sc, raw):
            chosen = raw
            break
    assert chosen is not None

    app_id = client.post("/api/applications/", json={"features": chosen},
                         headers=applicant_headers).json()["id"]
    p = client.post(f"/api/predictions/{app_id}", headers=applicant_headers)
    assert p.status_code == 200, p.text
    assert p.json()["decision"] == "REJECT"

    d = client.get(f"/api/applications/{app_id}", headers=applicant_headers)
    assert d.status_code == 200, d.text
    cfs = d.json()["explanation"]["counterfactuals"]
    assert cfs and len(cfs) <= 3
    for cf in cfs:
        assert cf["new_decision"] in ("REVIEW", "APPROVE")
        assert cf["changes"]

def test_review_application_gets_path_to_approve_with_drivers(client, applicant_headers):
    from backend.services.prediction_service import get_prediction_service
    from src.decision import decide
    from src.recourse import find_recourse

    sc = get_prediction_service().scorer
    base = pd.read_csv("data/processed/X_test.csv").head(600)
    fe = pd.read_csv("data/processed/X_test_fe_v2.csv").head(600)
    dec = decide(sc.model.predict(fe[sc.features]), sc.approve_below, sc.reject_from)

    chosen = None
    for i in np.where(dec == "review")[0][:60]:
        raw = json.loads(base.iloc[i].to_json())
        if find_recourse(sc, raw):
            chosen = raw
            break
    assert chosen is not None

    app_id = client.post("/api/applications/", json={"features": chosen},
                         headers=applicant_headers).json()["id"]
    p = client.post(f"/api/predictions/{app_id}", headers=applicant_headers)
    assert p.status_code == 200, p.text
    assert p.json()["decision"] == "REVIEW"

    expl = client.get(f"/api/applications/{app_id}", headers=applicant_headers).json()["explanation"]
    cfs = expl["counterfactuals"]
    assert cfs
    for cf in cfs:
        assert cf["new_decision"] == "APPROVE"
        assert cf["robust_margin"] > 0
        assert cf["drivers"] and {"feature", "delta"} <= set(cf["drivers"][0])
    lime_rows = expl["lime_explanation"]
    assert lime_rows and lime_rows[0]["rule"]


def _rejected_raw(sc):
    from src.decision import decide
    base = pd.read_csv("data/processed/X_test.csv").head(600)
    fe = pd.read_csv("data/processed/X_test_fe_v2.csv").head(600)
    dec = decide(sc.model.predict(fe[sc.features]), sc.approve_below, sc.reject_from)
    return json.loads(base.iloc[int((dec == "reject").argmax())].to_json())


def test_preferences_stored_and_validated(client, applicant_headers):
    prefs = {"locked_levers": ["income"], "max_loan_reduction": 0.2, "max_wait_months": 6}
    r = client.post("/api/applications/", json={"features": {"AMT_CREDIT": 100000},
                                                "recourse_preferences": prefs}, headers=applicant_headers)
    assert r.status_code == 201, r.text
    detail = client.get(f"/api/applications/{r.json()['id']}", headers=applicant_headers).json()
    assert detail["recourse_preferences"] == prefs

    for bad in ({"locked_levers": ["savings"]}, {"max_loan_reduction": 0.9}, {"max_wait_months": 5}):
        r = client.post("/api/applications/", json={"features": {"AMT_CREDIT": 100000},
                                                    "recourse_preferences": bad}, headers=applicant_headers)
        assert r.status_code == 422


def test_stored_preferences_shape_recourse(client, applicant_headers):
    from backend.services.prediction_service import get_prediction_service
    raw = _rejected_raw(get_prediction_service().scorer)
    prefs = {"locked_levers": ["income", "wait"], "max_loan_reduction": 0.5, "max_wait_months": 12}
    app_id = client.post("/api/applications/", json={"features": raw, "recourse_preferences": prefs},
                         headers=applicant_headers).json()["id"]
    assert client.post(f"/api/predictions/{app_id}", headers=applicant_headers).status_code == 200
    cfs = client.get(f"/api/applications/{app_id}", headers=applicant_headers).json()["explanation"]["counterfactuals"]
    for cf in cfs or []:
        assert "income" not in cf["levers"] and "wait" not in cf["levers"]


def test_recourse_endpoint_with_new_preferences(client, applicant_headers, make_applicant):
    from backend.services.prediction_service import get_prediction_service
    raw = _rejected_raw(get_prediction_service().scorer)
    app_id = client.post("/api/applications/", json={"features": raw}, headers=applicant_headers).json()["id"]

    url = f"/api/predictions/{app_id}/recourse"
    assert client.post(url, json={}, headers=applicant_headers).status_code == 400   # not scored yet
    assert client.post(f"/api/predictions/{app_id}", headers=applicant_headers).status_code == 200

    r = client.post(url, json={"locked_levers": ["income"], "max_wait_months": 3}, headers=applicant_headers)
    assert r.status_code == 200, r.text
    for cf in r.json():
        assert "income" not in cf["levers"]
        assert cf["wait_months"] <= 3

    assert client.post(url, json={"max_wait_months": 5}, headers=applicant_headers).status_code == 422
    other = make_applicant("other@test.com")
    assert client.post(url, json={}, headers=other).status_code == 403
    assert client.post("/api/predictions/999/recourse", json={}, headers=applicant_headers).status_code == 404


def test_recourse_endpoint_refuses_approved(client, applicant_headers, scored_application):
    app_id = scored_application(band="APPROVE")
    r = client.post(f"/api/predictions/{app_id}/recourse", json={}, headers=applicant_headers)
    assert r.status_code == 400
