import pytest


def _act(client, headers, app_id, action, justification=None):
    body = {"action": action}
    if justification is not None:
        body["justification"] = justification
    return client.post(f"/api/officer/{app_id}/action", json=body, headers=headers)


def test_queue_shows_full_details(client, applicant_headers, officer_headers, scored_application):
    app_id = scored_application(band="REVIEW")
    scored_application(band="APPROVE")
    queue = client.get("/api/officer/queue", headers=officer_headers)
    assert queue.status_code == 200, queue.text
    rows = queue.json()
    assert [r["id"] for r in rows] == [app_id]
    assert rows[0]["decision"]["decision"] == "REVIEW"
    assert rows[0]["features"]["AMT_CREDIT"] == 200000


def test_override_needs_justification(client, applicant_headers, officer_headers, scored_application):
    app_id = scored_application(band="REVIEW")
    assert _act(client, officer_headers, app_id, "approve").status_code == 400
    assert _act(client, officer_headers, app_id, "approve", "   ").status_code == 400


def test_approve_creates_loan_terms_over_chosen_tenure(client, applicant_headers, officer_headers,
                                                       scored_application):
    app_id = scored_application(band="REVIEW", tenure_months=36)
    r = _act(client, officer_headers, app_id, "approve", "Verified salary slips")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["application"]["status"] == "approved"
    terms = body["loan_terms"]
    assert terms["tenure_months"] == 36
    assert terms["loan_amount"] == 200000
    assert terms["monthly_emi"] * 36 == pytest.approx(terms["total_payable"], rel=1e-3)

    detail = client.get(f"/api/applications/{app_id}", headers=applicant_headers).json()
    assert detail["loan_terms"]["tenure_months"] == 36


def test_override_is_priced_at_model_band(client, applicant_headers, officer_headers, scored_application):
    review = scored_application(band="REVIEW")
    approve = scored_application(band="APPROVE")
    r_rate = _act(client, officer_headers, review, "approve", "ok").json()["loan_terms"]["interest_rate"]
    a_rate = _act(client, officer_headers, approve, "approve").json()["loan_terms"]["interest_rate"]
    assert r_rate > a_rate


def test_agreeing_with_model_needs_no_justification(client, applicant_headers, officer_headers,
                                                    scored_application):
    assert _act(client, officer_headers, scored_application(band="APPROVE"), "approve").status_code == 200
    assert _act(client, officer_headers, scored_application(band="REJECT"), "reject").status_code == 200


def test_rejecting_an_approve_case_needs_justification(client, applicant_headers, officer_headers,
                                                       scored_application):
    app_id = scored_application(band="APPROVE")
    assert _act(client, officer_headers, app_id, "reject").status_code == 400
    assert _act(client, officer_headers, app_id, "reject", "Forged document").status_code == 200


def test_request_documents_needs_message(client, applicant_headers, officer_headers, scored_application):
    app_id = scored_application(band="REVIEW")
    assert _act(client, officer_headers, app_id, "request_documents").status_code == 400
    r = _act(client, officer_headers, app_id, "request_documents", "Bank statement")
    assert r.status_code == 200
    assert r.json()["application"]["status"] == "under_review"


def test_invalid_or_unscored(client, applicant_headers, officer_headers, scored_application):
    assert _act(client, officer_headers, scored_application(band="REVIEW"), "delete", "x").status_code == 400
    unscored = client.post("/api/applications/", json={"features": {"AMT_CREDIT": 1000}},
                           headers=applicant_headers).json()["id"]
    assert _act(client, officer_headers, unscored, "approve", "x").status_code == 400
    assert _act(client, officer_headers, 999, "approve", "x").status_code == 404


def test_audit_log(client, applicant_headers, officer_headers, scored_application):
    app_id = scored_application(band="REVIEW")
    _act(client, officer_headers, app_id, "request_documents", "ID proof")
    _act(client, officer_headers, app_id, "approve", "Documents fine")
    log = client.get(f"/api/officer/{app_id}/audit-log", headers=officer_headers)
    assert log.status_code == 200
    actions = [e["action"] for e in log.json()]
    assert set(actions) == {"request_documents", "approve"}
    assert client.get("/api/officer/999/audit-log", headers=officer_headers).status_code == 404


def test_approval_rate_analytics(client, applicant_headers, officer_headers, scored_application):
    scored_application(band="APPROVE")
    scored_application(band="REVIEW")
    stats = client.get("/api/officer/analytics/approval-rates", headers=officer_headers).json()
    assert stats["total_applications"] == 2
    assert stats["approved"] == 1 and stats["under_review"] == 1
    assert stats["approval_rate"] == pytest.approx(0.5)


def test_applicant_cannot_act(client, applicant_headers, scored_application):
    app_id = scored_application(band="REVIEW")
    assert _act(client, applicant_headers, app_id, "approve", "me").status_code == 403
