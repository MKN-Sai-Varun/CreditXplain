PDF = b"%PDF-1.4\n%test document\n"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _new_app(client, headers):
    r = client.post("/api/applications/", json={"features": {"AMT_CREDIT": 150000}}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _upload(client, app_id, headers, content=PDF, name="payslip.pdf", ctype="application/pdf",
            doc_type="income_proof"):
    return client.post(f"/api/applications/{app_id}/documents",
                       files={"file": (name, content, ctype)},
                       data={"doc_type": doc_type}, headers=headers)


def test_upload_list_and_download(client, applicant_headers, upload_dir):
    app_id = _new_app(client, applicant_headers)
    r = _upload(client, app_id, applicant_headers)
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["original_name"] == "payslip.pdf"
    assert doc["size_bytes"] == len(PDF)
    assert len(list((upload_dir / str(app_id)).iterdir())) == 1

    listing = client.get(f"/api/applications/{app_id}/documents", headers=applicant_headers).json()
    assert [d["id"] for d in listing["documents"]] == [doc["id"]]

    dl = client.get(f"/api/applications/{app_id}/documents/{doc['id']}", headers=applicant_headers)
    assert dl.status_code == 200
    assert dl.content == PDF
    assert dl.headers["content-type"].startswith("application/pdf")

    detail = client.get(f"/api/applications/{app_id}", headers=applicant_headers).json()
    assert detail["documents"][0]["id"] == doc["id"]


def test_png_accepted(client, applicant_headers):
    app_id = _new_app(client, applicant_headers)
    assert _upload(client, app_id, applicant_headers, PNG, "id.png", "image/png", "id_proof").status_code == 201


def test_wrong_type_refused(client, applicant_headers):
    app_id = _new_app(client, applicant_headers)
    assert _upload(client, app_id, applicant_headers, b"hello", "a.txt", "text/plain").status_code == 415


def test_content_must_match_declared_type(client, applicant_headers):
    app_id = _new_app(client, applicant_headers)
    assert _upload(client, app_id, applicant_headers, b"not really a pdf").status_code == 415


def test_too_large_refused(client, applicant_headers, monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 10)
    app_id = _new_app(client, applicant_headers)
    assert _upload(client, app_id, applicant_headers).status_code == 413


def test_empty_and_unknown_doc_type_refused(client, applicant_headers):
    app_id = _new_app(client, applicant_headers)
    assert _upload(client, app_id, applicant_headers, b"").status_code in (400, 415)
    assert _upload(client, app_id, applicant_headers, doc_type="selfie").status_code == 400


def test_other_applicant_has_no_access(client, make_applicant):
    a = make_applicant("a1@test.com")
    b = make_applicant("b1@test.com")
    app_id = _new_app(client, a)
    doc_id = _upload(client, app_id, a).json()["id"]
    assert _upload(client, app_id, b).status_code == 403
    assert client.get(f"/api/applications/{app_id}/documents", headers=b).status_code == 403
    assert client.get(f"/api/applications/{app_id}/documents/{doc_id}", headers=b).status_code == 403


def test_officer_can_read_but_not_upload(client, applicant_headers, officer_headers):
    app_id = _new_app(client, applicant_headers)
    doc_id = _upload(client, app_id, applicant_headers).json()["id"]
    assert client.get(f"/api/applications/{app_id}/documents", headers=officer_headers).status_code == 200
    dl = client.get(f"/api/applications/{app_id}/documents/{doc_id}", headers=officer_headers)
    assert dl.status_code == 200 and dl.content == PDF
    assert _upload(client, app_id, officer_headers).status_code == 403


def test_missing_application_or_document(client, applicant_headers):
    assert client.get("/api/applications/999/documents", headers=applicant_headers).status_code == 404
    app_id = _new_app(client, applicant_headers)
    assert client.get(f"/api/applications/{app_id}/documents/999", headers=applicant_headers).status_code == 404


def test_applicant_sees_officer_request(client, applicant_headers, officer_headers, scored_application):
    app_id = scored_application(band="REVIEW")
    r = client.post(f"/api/officer/{app_id}/action",
                    json={"action": "request_documents", "justification": "Last 3 months of payslips"},
                    headers=officer_headers)
    assert r.status_code == 200, r.text
    listing = client.get(f"/api/applications/{app_id}/documents", headers=applicant_headers).json()
    assert listing["requests"][0]["message"] == "Last 3 months of payslips"
    detail = client.get(f"/api/applications/{app_id}", headers=applicant_headers).json()
    assert detail["document_requests"][0]["message"] == "Last 3 months of payslips"


def test_upload_is_recorded_in_audit_log(client, applicant_headers, officer_headers):
    app_id = _new_app(client, applicant_headers)
    _upload(client, app_id, applicant_headers)
    log = client.get(f"/api/officer/{app_id}/audit-log", headers=officer_headers).json()
    assert log[0]["action"] == "document_uploaded"
    assert log[0]["officer_id"] is None
