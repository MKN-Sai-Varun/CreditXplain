from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator
from datetime import datetime
from typing import Optional, Dict, List, Any
from backend.config import settings
from backend.models import UserRole, ApplicationStatus, DecisionType

# User schemas
class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(UserBase):
    id: int
    role: UserRole
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

# Token schemas
class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class TokenData(BaseModel):
    email: Optional[str] = None

# Recourse preferences
class RecoursePreferences(BaseModel):
    locked_levers: List[str] = Field([], description="Levers the applicant cannot use: credit, income, tenure, wait")
    max_loan_reduction: float = Field(settings.MAX_LOAN_REDUCTION, description="Largest share the loan may shrink by")
    max_wait_months: int = Field(max(settings.WAIT_OPTIONS), description="Longest acceptable wait before re-applying")

    @field_validator("locked_levers")
    @classmethod
    def known_levers(cls, v):
        unknown = set(v) - set(settings.RECOURSE_LEVERS)
        if unknown:
            raise ValueError(f"unknown levers {sorted(unknown)}; choose from {settings.RECOURSE_LEVERS}")
        return sorted(set(v))

    @field_validator("max_loan_reduction")
    @classmethod
    def reduction_range(cls, v):
        if not 0 <= v <= settings.MAX_LOAN_REDUCTION:
            raise ValueError(f"max_loan_reduction must be between 0 and {settings.MAX_LOAN_REDUCTION}")
        return v

    @field_validator("max_wait_months")
    @classmethod
    def wait_option(cls, v):
        if v not in settings.WAIT_OPTIONS:
            raise ValueError(f"max_wait_months must be one of {settings.WAIT_OPTIONS}")
        return v

    def as_kwargs(self):
        return {"locked": tuple(self.locked_levers), "max_loan_reduction": self.max_loan_reduction,
                "max_wait_months": self.max_wait_months}


# Application schemas
class ApplicationCreate(BaseModel):
    features: Dict[str, Any] = Field(..., description="Application features as key-value pairs")
    tenure_months: int = Field(settings.DEFAULT_TENURE_MONTHS, description="Requested repayment period in months")
    recourse_preferences: Optional[RecoursePreferences] = None

    @field_validator("features")
    @classmethod
    def loan_amount_required(cls, v):
        amount = v.get("AMT_CREDIT")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount <= 0:
            raise ValueError("features.AMT_CREDIT (requested loan amount) must be a positive number")
        return v

    @field_validator("tenure_months")
    @classmethod
    def tenure_allowed(cls, v):
        if v not in settings.TENURE_OPTIONS:
            raise ValueError(f"tenure_months must be one of {settings.TENURE_OPTIONS}")
        return v

class ApplicationResponse(BaseModel):
    id: int
    user_id: int
    status: ApplicationStatus
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

# Decision schemas
class DecisionResponse(BaseModel):
    id: int
    application_id: int
    decision: DecisionType
    probability: float
    approve_threshold: float
    reject_threshold: float
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

# Explanation schemas
class ShapValue(BaseModel):
    feature: str
    value: float
    feature_value: Any
    label: Optional[str] = None

class LimeFeature(BaseModel):
    feature: str
    weight: float
    rule: Optional[str] = None
    label: Optional[str] = None

class Counterfactual(BaseModel):
    levers: List[str] = []
    changes: List[Dict[str, Any]]
    num_changes: int
    wait_months: int = 0
    original_probability: Optional[float] = None
    new_probability: float
    new_decision: str
    reaches_approve: bool = False
    success: bool = True
    cost: Optional[float] = None
    feasibility_score: Optional[float] = None
    robust_margin: Optional[float] = None
    plausible: Optional[bool] = None
    plausibility_distance: Optional[float] = None
    drivers: Optional[List[Dict[str, Any]]] = None
    retrain_check_passed: Optional[bool] = None
    ensemble_agreement: Optional[str] = None

class ExplanationResponse(BaseModel):
    shap_values: List[ShapValue]
    lime_explanation: List[LimeFeature]
    counterfactuals: Optional[List[Counterfactual]] = None
    
    model_config = ConfigDict(from_attributes=True)

# Loan terms schemas
class LoanTermsResponse(BaseModel):
    loan_amount: float
    interest_rate: float
    tenure_months: int
    monthly_emi: float
    total_payable: float
    
    model_config = ConfigDict(from_attributes=True)

# Documents
class DocumentResponse(BaseModel):
    id: int
    application_id: int
    doc_type: str
    original_name: str
    content_type: str
    size_bytes: int
    uploaded_at: datetime

    model_config = ConfigDict(from_attributes=True)

class DocumentRequest(BaseModel):
    message: Optional[str]
    requested_at: datetime

class DocumentsOverview(BaseModel):
    requests: List[DocumentRequest]
    documents: List[DocumentResponse]

# Full application detail
class ApplicationDetail(ApplicationResponse):
    features: Dict[str, Any]
    tenure_months: int = settings.DEFAULT_TENURE_MONTHS
    recourse_preferences: Optional[RecoursePreferences] = None
    decision: Optional[DecisionResponse] = None
    explanation: Optional[ExplanationResponse] = None
    loan_terms: Optional[LoanTermsResponse] = None
    documents: List[DocumentResponse] = []
    document_requests: List[DocumentRequest] = []

# Officer schemas
class OfficerAction(BaseModel):
    action: str = Field(..., description="approve, reject, request_documents")
    justification: Optional[str] = None

class OfficerActionResponse(BaseModel):
    message: str
    application: ApplicationResponse
    loan_terms: Optional[LoanTermsResponse] = None

class AuditLogResponse(BaseModel):
    id: int
    application_id: int
    officer_id: Optional[int]
    action: str
    justification: Optional[str]
    previous_status: Optional[str]
    new_status: Optional[str]
    timestamp: datetime
    
    model_config = ConfigDict(from_attributes=True)

# Fairness schemas
class FairnessMetricsResponse(BaseModel):
    age_dpd: Optional[float]
    children_dpd: Optional[float]
    has_alert: bool
    alert_message: Optional[str]
    metrics_detail: Dict[str, Any]
    calculated_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)

# Analytics schemas
class ApprovalRateStats(BaseModel):
    total_applications: int
    approved: int
    rejected: int
    under_review: int
    approval_rate: float

class DemographicStats(BaseModel):
    group: str
    count: int
    approved: int
    rejected: int
    approval_rate: float
