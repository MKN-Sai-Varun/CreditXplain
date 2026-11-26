from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from backend.database import Base

class UserRole(str, enum.Enum):
    APPLICANT = "applicant"
    OFFICER = "officer"
    ADMIN = "admin"

class ApplicationStatus(str, enum.Enum):
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"

class DecisionType(str, enum.Enum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    REJECT = "REJECT"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.APPLICANT, nullable=False)
    full_name = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    applications = relationship("Application", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="officer")

class Application(Base):
    __tablename__ = "applications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Application data (JSON for flexibility)
    features_json = Column(JSON, nullable=False)
    
    # Requested repayment period
    tenure_months = Column(Integer, default=12, nullable=False)

    # Status
    status = Column(Enum(ApplicationStatus), default=ApplicationStatus.SUBMITTED, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="applications")
    decision = relationship("Decision", back_populates="application", uselist=False)
    audit_logs = relationship("AuditLog", back_populates="application")
    documents = relationship("Document", back_populates="application", order_by="Document.uploaded_at")

class Decision(Base):
    __tablename__ = "decisions"
    
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("applications.id"), unique=True, nullable=False)
    
    # Decision details
    decision = Column(Enum(DecisionType), nullable=False)
    probability = Column(Float, nullable=False)
    approve_threshold = Column(Float, nullable=False)
    reject_threshold = Column(Float, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    application = relationship("Application", back_populates="decision")
    explanation = relationship("Explanation", back_populates="decision", uselist=False)
    loan_terms = relationship("LoanTerms", back_populates="decision", uselist=False)

class Explanation(Base):
    __tablename__ = "explanations"
    
    id = Column(Integer, primary_key=True, index=True)
    decision_id = Column(Integer, ForeignKey("decisions.id"), unique=True, nullable=False)
    
    # Explanation data
    shap_values = Column(JSON)  # SHAP feature importance
    lime_explanation = Column(JSON)  # LIME local explanation
    counterfactuals = Column(JSON)  # Counterfactual recommendations
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    decision = relationship("Decision", back_populates="explanation")

class LoanTerms(Base):
    __tablename__ = "loan_terms"
    
    id = Column(Integer, primary_key=True, index=True)
    decision_id = Column(Integer, ForeignKey("decisions.id"), unique=True, nullable=False)
    
    # Loan details
    loan_amount = Column(Float, nullable=False)
    interest_rate = Column(Float, nullable=False)  # Annual rate
    tenure_months = Column(Integer, nullable=False)
    monthly_emi = Column(Float, nullable=False)
    total_payable = Column(Float, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    decision = relationship("Decision", back_populates="loan_terms")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("applications.id"), nullable=False)
    officer_id = Column(Integer, ForeignKey("users.id"))
    
    # Action details
    action = Column(String(50), nullable=False)  # "approved", "rejected", "requested_documents"
    justification = Column(Text)
    previous_status = Column(String(50))
    new_status = Column(String(50))
    
    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    application = relationship("Application", back_populates="audit_logs")
    officer = relationship("User", back_populates="audit_logs")

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("applications.id"), nullable=False, index=True)
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=False)

    doc_type = Column(String(50), nullable=False)        # e.g. "income_proof", "id_proof"
    original_name = Column(String(255), nullable=False)
    stored_name = Column(String(64), unique=True, nullable=False)
    content_type = Column(String(100), nullable=False)
    size_bytes = Column(Integer, nullable=False)

    uploaded_at = Column(DateTime, default=datetime.utcnow)

    application = relationship("Application", back_populates="documents")


class FairnessMetric(Base):
    __tablename__ = "fairness_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Metrics
    age_dpd = Column(Float)  # Demographic Parity Difference for age
    children_dpd = Column(Float)  # DPD for family status
    metrics_json = Column(JSON)  # Full metrics breakdown
    
    # Alert flags
    has_alert = Column(Integer, default=0)  # 1 if any metric exceeds threshold
    alert_message = Column(Text)
    
    # Timestamps
    calculated_at = Column(DateTime, default=datetime.utcnow)
