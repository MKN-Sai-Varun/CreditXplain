import pytest


def test_metrics_before_any_calculation(client, officer_headers):
    r = client.get("/api/fairness/metrics", headers=officer_headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["calculated_at"] is None
    assert body["has_alert"] is False


def test_calculate_needs_fifty_applications(client, applicant_headers, officer_headers, scored_application):
    for _ in range(5):
        scored_application(band="APPROVE")
    r = client.post("/api/fairness/calculate", headers=officer_headers)
    assert r.status_code == 200
    assert r.json()["success"] is False


@pytest.fixture
def sixty_scored(applicant_headers, scored_application):
    # younger applicants approved less often than older ones, so an alert fires
    for i in range(60):
        young = i % 2 == 0
        band = "REVIEW" if (young and i % 4 == 0) else "APPROVE"
        days_birth = -22 * 365 if young else -50 * 365
        scored_application(band=band, features={"AMT_CREDIT": 100000, "DAYS_BIRTH": days_birth,
                                                "CNT_CHILDREN": i % 3})


def test_calculate_and_read_back(client, officer_headers, sixty_scored):
    r = client.post("/api/fairness/calculate", headers=officer_headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["metrics"]["total_applications"] == 60

    m = client.get("/api/fairness/metrics", headers=officer_headers).json()
    assert m["calculated_at"] is not None
    assert m["age_dpd"] == pytest.approx(0.5, abs=1e-6)   # <25: 50% approved, 45-55: 100%
    assert m["has_alert"] is True


def test_demographic_breakdowns(client, officer_headers, sixty_scored):
    client.post("/api/fairness/calculate", headers=officer_headers)

    age = client.get("/api/fairness/demographics/age", headers=officer_headers)
    assert age.status_code == 200, age.text
    groups = {g["group"]: g for g in age.json()}
    assert groups["<25"]["count"] == 30 and groups["<25"]["approved"] == 15
    assert groups["45-55"]["approval_rate"] == pytest.approx(1.0)
    assert sum(g["count"] for g in groups.values()) == 60

    fam = client.get("/api/fairness/demographics/family", headers=officer_headers)
    assert fam.status_code == 200, fam.text
    assert sum(g["count"] for g in fam.json()) == 60
    for g in fam.json():
        assert 0.0 <= g["approval_rate"] <= 1.0


def test_demographics_empty_before_calculation(client, officer_headers):
    assert client.get("/api/fairness/demographics/age", headers=officer_headers).json() == []
