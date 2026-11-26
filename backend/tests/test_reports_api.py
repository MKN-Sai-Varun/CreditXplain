"""
Decision and compliance PDF reports. PDFs are uncompressed, so their text can be
checked directly in the bytes.
"""
import json
import time
from pathlib import Path

import pytest


def _pdf(r):
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/pdf")
    assert r.content.startswith(b"%PDF")
    return r.content


def test_applicant_report_for_scored_application(client, applicant_headers, scored_application):
    app_id = scored_application(band="REVIEW")
    t = time.perf_counter()
    pdf = _pdf(client.get(f"/api/applications/{app_id}/report.pdf", headers=applicant_headers))
    assert time.perf_counter() - t < 2.0
    assert b"Credit Decision Report" in pdf
    assert b"REVIEW" in pdf
    assert b"Main reasons" in pdf
    assert b"No explanation is available" in pdf      # fixture stores no explanation
    assert b"attachment" in client.get(f"/api/applications/{app_id}/report.pdf",
                                       headers=applicant_headers).headers["content-disposition"].encode()


def test_applicant_report_access_and_errors(client, make_applicant, officer_headers, scored_application):
    owner = make_applicant("a@test.com")
    app_id = scored_application(band="REJECT")
    other = make_applicant("b@test.com")
    assert client.get(f"/api/applications/{app_id}/report.pdf", headers=other).status_code == 403
    assert client.get(f"/api/applications/{app_id}/report.pdf", headers=officer_headers).status_code == 200
    assert client.get("/api/applications/999/report.pdf", headers=owner).status_code == 404

    unscored = client.post("/api/applications/", json={"features": {"AMT_CREDIT": 1000}}, headers=owner).json()["id"]
    assert client.get(f"/api/applications/{unscored}/report.pdf", headers=owner).status_code == 400


def test_report_shows_loan_terms_and_document_requests(client, applicant_headers, officer_headers,
                                                      scored_application):
    app_id = scored_application(band="REVIEW", tenure_months=24)
    client.post(f"/api/officer/{app_id}/action",
                json={"action": "request_documents", "justification": "Payslips"}, headers=officer_headers)
    client.post(f"/api/officer/{app_id}/action",
                json={"action": "approve", "justification": "Verified"}, headers=officer_headers)
    pdf = _pdf(client.get(f"/api/applications/{app_id}/report.pdf", headers=applicant_headers))
    assert b"Loan terms" in pdf
    assert b"24 months" in pdf
    assert b"Documents requested" in pdf and b"Payslips" in pdf
    assert b"approved" in pdf                           # status after officer review


def test_compliance_report_officer_only(client, applicant_headers, officer_headers, scored_application):
    app_id = scored_application(band="REVIEW")
    assert client.get(f"/api/officer/{app_id}/compliance-report.pdf", headers=applicant_headers).status_code == 403
    assert client.get("/api/officer/999/compliance-report.pdf", headers=officer_headers).status_code == 404


def test_compliance_report_content_and_audit(client, applicant_headers, officer_headers, scored_application):
    app_id = scored_application(band="REVIEW")
    client.post(f"/api/officer/{app_id}/action",
                json={"action": "approve", "justification": "Verified"}, headers=officer_headers)
    pdf = _pdf(client.get(f"/api/officer/{app_id}/compliance-report.pdf", headers=officer_headers))
    for text in (b"Compliance Report", b"Audit trail", b"Verified", b"Recourse preferences",
                 b"No documents uploaded", b"No fairness calculation"):
        assert text in pdf, text
    if Path("models/final_model_card.json").exists():
        assert b"LightGBM" in pdf

    log = client.get(f"/api/officer/{app_id}/audit-log", headers=officer_headers).json()
    assert log[0]["action"] == "compliance_report"


def test_compliance_report_unscored(client, applicant_headers, officer_headers):
    app_id = client.post("/api/applications/", json={"features": {"AMT_CREDIT": 1000}},
                         headers=applicant_headers).json()["id"]
    assert client.get(f"/api/officer/{app_id}/compliance-report.pdf", headers=officer_headers).status_code == 400


DATA = ["models/final_lgbm.pkl", "data/processed/X_test.csv", "data/processed/X_test_fe_v2.csv"]


@pytest.mark.skipif(not all(Path(p).exists() for p in DATA), reason="model/data missing")
def test_report_after_real_prediction_lists_reasons_and_recourse(client, applicant_headers, officer_headers):
    import numpy as np
    import pandas as pd
    from backend.services.prediction_service import get_prediction_service
    from src.decision import decide
    from src.recourse import find_recourse

    sc = get_prediction_service().scorer
    base = pd.read_csv("data/processed/X_test.csv").head(600)
    fe = pd.read_csv("data/processed/X_test_fe_v2.csv").head(600)
    dec = decide(sc.model.predict(fe[sc.features]), sc.approve_below, sc.reject_from)
    raw = next(json.loads(base.iloc[i].to_json()) for i in np.where(dec == "reject")[0][:40]
               if find_recourse(sc, base.iloc[i].to_dict()))

    app_id = client.post("/api/applications/", json={"features": raw}, headers=applicant_headers).json()["id"]
    assert client.post(f"/api/predictions/{app_id}", headers=applicant_headers).status_code == 200

    pdf = _pdf(client.get(f"/api/applications/{app_id}/report.pdf", headers=applicant_headers))
    assert b"REJECT" in pdf
    assert b"raises risk" in pdf or b"lowers risk" in pdf
    assert b"What could change the decision" in pdf
    assert b"Option 1" in pdf

    comp = _pdf(client.get(f"/api/officer/{app_id}/compliance-report.pdf", headers=officer_headers))
    assert b"SHAP contributions" in comp
    assert b"LIME local rules" in comp
