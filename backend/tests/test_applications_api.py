import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import StaticPool


def _create(client, headers, features, **extra):
    return client.post("/api/applications/", json={"features": features, **extra}, headers=headers)


def test_tenure_defaults_to_twelve(client, applicant_headers):
    app_id = _create(client, applicant_headers, {"AMT_CREDIT": 100000}).json()["id"]
    assert client.get(f"/api/applications/{app_id}", headers=applicant_headers).json()["tenure_months"] == 12


def test_tenure_is_stored(client, applicant_headers):
    r = _create(client, applicant_headers, {"AMT_CREDIT": 100000}, tenure_months=48)
    assert r.status_code == 201, r.text
    detail = client.get(f"/api/applications/{r.json()['id']}", headers=applicant_headers).json()
    assert detail["tenure_months"] == 48


@pytest.mark.parametrize("tenure", [0, 18, 61, 120])
def test_tenure_must_be_an_offered_option(client, applicant_headers, tenure):
    assert _create(client, applicant_headers, {"AMT_CREDIT": 100000}, tenure_months=tenure).status_code == 422


@pytest.mark.parametrize("features", [{}, {"AMT_CREDIT": 0}, {"AMT_CREDIT": -5},
                                      {"AMT_CREDIT": "lots"}, {"AMT_CREDIT": True}])
def test_loan_amount_required(client, applicant_headers, features):
    assert _create(client, applicant_headers, features).status_code == 422


def test_old_database_gets_tenure_column():
    from backend.database import init_db

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE applications (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, "
            "features_json JSON NOT NULL, status VARCHAR(12) NOT NULL, "
            "created_at DATETIME, updated_at DATETIME)"))
        conn.execute(text("INSERT INTO applications (id, user_id, features_json, status) "
                          "VALUES (1, 1, '{}', 'SUBMITTED')"))

    init_db(engine)
    init_db(engine)   # running twice must be harmless

    cols = {c["name"] for c in inspect(engine).get_columns("applications")}
    assert "tenure_months" in cols
    assert inspect(engine).has_table("documents")
    with engine.connect() as conn:
        assert conn.execute(text("SELECT tenure_months FROM applications WHERE id = 1")).scalar() == 12
