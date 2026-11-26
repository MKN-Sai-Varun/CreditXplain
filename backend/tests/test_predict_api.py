from pathlib import Path
import pytest

pytestmark = pytest.mark.skipif(
    not Path("models/final_lgbm.pkl").exists(), reason="model not trained"
)

FEATURES = {
    "AMT_CREDIT": 500000, "AMT_INCOME_TOTAL": 180000, "AMT_GOODS_PRICE": 450000,
    "AMT_ANNUITY": 25000, "DAYS_BIRTH": -12000, "EXT_SOURCE_2": 0.55, "EXT_SOURCE_3": 0.5,
}


def test_predict_then_read_back(client, applicant_headers):
    r = client.post("/api/applications/", json={"features": FEATURES}, headers=applicant_headers)
    app_id = r.json()["id"]

    p = client.post(f"/api/predictions/{app_id}", headers=applicant_headers)
    assert p.status_code == 200, p.text
    body = p.json()
    assert 0 <= body["probability"] <= 1
    assert body["decision"] in ("APPROVE", "REVIEW", "REJECT")

    d = client.get(f"/api/applications/{app_id}", headers=applicant_headers)
    assert d.status_code == 200, d.text  # fails if the response schemas mismatch
    shap_rows = d.json()["explanation"]["shap_values"]
    assert len(shap_rows) > 0
    assert {"feature", "value", "feature_value"} <= set(shap_rows[0])


def test_cannot_predict_twice(client, applicant_headers):
    app_id = client.post("/api/applications/", json={"features": FEATURES},
                         headers=applicant_headers).json()["id"]
    assert client.post(f"/api/predictions/{app_id}", headers=applicant_headers).status_code == 200
    assert client.post(f"/api/predictions/{app_id}", headers=applicant_headers).status_code == 400

def test_approved_prediction_uses_requested_tenure(client, applicant_headers):
    import json
    import pandas as pd
    if not (Path("data/processed/X_test.csv").exists() and Path("data/processed/X_test_fe_v2.csv").exists()):
        pytest.skip("test data missing")
    from backend.services.prediction_service import get_prediction_service
    from src.decision import decide

    sc = get_prediction_service().scorer
    base = pd.read_csv("data/processed/X_test.csv").head(300)
    fe = pd.read_csv("data/processed/X_test_fe_v2.csv").head(300)
    dec = decide(sc.model.predict(fe[sc.features]), sc.approve_below, sc.reject_from)
    raw = json.loads(base.iloc[int((dec == "approve").argmax())].to_json())

    app_id = client.post("/api/applications/", json={"features": raw, "tenure_months": 48},
                         headers=applicant_headers).json()["id"]
    p = client.post(f"/api/predictions/{app_id}", headers=applicant_headers)
    assert p.status_code == 200, p.text
    assert p.json()["decision"] == "APPROVE"
    terms = client.get(f"/api/applications/{app_id}", headers=applicant_headers).json()["loan_terms"]
    assert terms["tenure_months"] == 48
    assert terms["loan_amount"] == pytest.approx(raw["AMT_CREDIT"])
