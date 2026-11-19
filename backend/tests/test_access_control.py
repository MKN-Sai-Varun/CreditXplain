def test_register_ignores_requested_role(client):
    r = client.post("/api/auth/register", json={
        "email": "evil@test.com", "password": "pw12345", "role": "officer"})
    assert r.status_code == 201
    assert r.json()["role"] == "applicant"


def test_self_registered_user_blocked_from_officer_routes(client, applicant_headers):
    assert client.get("/api/officer/queue", headers=applicant_headers).status_code == 403
    assert client.get("/api/fairness/metrics", headers=applicant_headers).status_code == 403
    assert client.post("/api/fairness/calculate", headers=applicant_headers).status_code == 403


def test_officer_can_reach_officer_routes(client, officer_headers):
    assert client.get("/api/officer/queue", headers=officer_headers).status_code == 200


def test_no_token_is_rejected(client):
    assert client.get("/api/officer/queue").status_code == 401
    assert client.get("/api/applications/").status_code == 401


def test_applicant_cannot_read_someone_elses_application(client, make_applicant):
    a = make_applicant("a1@test.com")
    b = make_applicant("b1@test.com")
    app_id = client.post("/api/applications/", json={"features": {"AMT_CREDIT": 1}},
                         headers=a).json()["id"]
    assert client.get(f"/api/applications/{app_id}", headers=b).status_code == 403