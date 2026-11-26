import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app
from backend.database import Base, get_db
from backend.models import User, UserRole
from backend.auth import get_password_hash

PASSWORD = "pw12345"


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


@pytest.fixture
def client(session_factory):
    def override():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    yield TestClient(app)
    app.dependency_overrides.clear()


def _login(client, email):
    r = client.post("/api/auth/login", data={"username": email, "password": PASSWORD})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def make_applicant(client):
    def _make(email="a@test.com"):
        client.post("/api/auth/register", json={"email": email, "password": PASSWORD, "full_name": "T"})
        return _login(client, email)
    return _make


@pytest.fixture
def applicant_headers(make_applicant):
    return make_applicant()


@pytest.fixture
def officer_headers(client, session_factory):
    db = session_factory()
    db.add(User(email="o@test.com", password_hash=get_password_hash(PASSWORD),
                role=UserRole.OFFICER, full_name="O"))
    db.commit()
    db.close()
    return _login(client, "o@test.com")

@pytest.fixture(autouse=True)
def upload_dir(tmp_path, monkeypatch):
    """Uploaded files go to a temporary folder, never the real uploads/."""
    from backend.config import settings
    d = tmp_path / "uploads"
    monkeypatch.setattr(settings, "UPLOAD_DIR", d)
    return d


@pytest.fixture
def scored_application(session_factory):
    """
    Creates an application with a stored model decision, without running the
    model. Returns the application id.
    """
    from backend.models import Application, ApplicationStatus, Decision, DecisionType, User

    status_for = {"APPROVE": ApplicationStatus.APPROVED, "REVIEW": ApplicationStatus.UNDER_REVIEW,
                  "REJECT": ApplicationStatus.REJECTED}
    probability_for = {"APPROVE": 0.02, "REVIEW": 0.07, "REJECT": 0.2}

    def _make(email="a@test.com", band="REVIEW", features=None, tenure_months=12):
        db = session_factory()
        user = db.query(User).filter(User.email == email).first()
        app = Application(user_id=user.id, tenure_months=tenure_months,
                          features_json=features or {"AMT_CREDIT": 200000, "DAYS_BIRTH": -12000},
                          status=status_for[band])
        db.add(app)
        db.flush()
        db.add(Decision(application_id=app.id, decision=DecisionType[band],
                        probability=probability_for[band], approve_threshold=0.0349,
                        reject_threshold=0.1189))
        db.commit()
        app_id = app.id
        db.close()
        return app_id

    return _make
