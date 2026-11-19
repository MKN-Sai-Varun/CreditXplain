import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.database import Base
from backend.models import User, Application, Decision, UserRole, ApplicationStatus, DecisionType

@pytest.fixture
def test_db():
    """Create an in-memory SQLite database for testing"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()

def test_create_user(test_db):
    """Test creating a user"""
    user = User(
        email="test@example.com",
        password_hash="hashed_password",
        role=UserRole.APPLICANT,
        full_name="Test User"
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    
    assert user.id is not None
    assert user.email == "test@example.com"
    assert user.role == UserRole.APPLICANT

def test_create_application(test_db):
    """Test creating an application"""
    user = User(
        email="test@example.com",
        password_hash="hashed_password",
        role=UserRole.APPLICANT
    )
    test_db.add(user)
    test_db.commit()
    
    application = Application(
        user_id=user.id,
        features_json={"AMT_CREDIT": 100000, "AMT_INCOME_TOTAL": 50000},
        status=ApplicationStatus.SUBMITTED
    )
    test_db.add(application)
    test_db.commit()
    test_db.refresh(application)
    
    assert application.id is not None
    assert application.user_id == user.id
    assert application.status == ApplicationStatus.SUBMITTED

def test_user_application_relationship(test_db):
    """Test relationship between user and applications"""
    user = User(
        email="test@example.com",
        password_hash="hashed_password",
        role=UserRole.APPLICANT
    )
    test_db.add(user)
    test_db.commit()
    
    app1 = Application(
        user_id=user.id,
        features_json={"test": 1},
        status=ApplicationStatus.SUBMITTED
    )
    app2 = Application(
        user_id=user.id,
        features_json={"test": 2},
        status=ApplicationStatus.SUBMITTED
    )
    test_db.add_all([app1, app2])
    test_db.commit()
    
    # Query user and check applications
    queried_user = test_db.query(User).filter(User.email == "test@example.com").first()
    assert len(queried_user.applications) == 2

def test_application_decision_relationship(test_db):
    """Test relationship between application and decision"""
    user = User(email="test@example.com", password_hash="hash", role=UserRole.APPLICANT)
    test_db.add(user)
    test_db.commit()
    
    application = Application(
        user_id=user.id,
        features_json={"test": 1},
        status=ApplicationStatus.SUBMITTED
    )
    test_db.add(application)
    test_db.commit()
    
    decision = Decision(
        application_id=application.id,
        decision=DecisionType.APPROVE,
        probability=0.02,
        approve_threshold=0.0349,
        reject_threshold=0.1189
    )
    test_db.add(decision)
    test_db.commit()
    
    # Query and verify relationship
    queried_app = test_db.query(Application).filter(Application.id == application.id).first()
    assert queried_app.decision is not None
    assert queried_app.decision.decision == DecisionType.APPROVE
