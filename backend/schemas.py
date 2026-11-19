from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional, Dict, List, Any
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
    
    class Config:
        from_attributes = True

# Token schemas
class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class TokenData(BaseModel):
    email: Optional[str] = None

# Application schemas
class ApplicationCreate(BaseModel):
    features: Dict[str, Any] = Field(..., description="Application features as key-value pairs")

class ApplicationResponse(BaseModel):
    id: int
    user_id: int
    status: ApplicationStatus
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Decision schemas
class DecisionResponse(BaseModel):
    id: int
    application_id: int
    decision: DecisionType
    probability: float
    approve_threshold: float
    reject_threshold: float
    created_at: datetime
    
    class Config:
        from_attributes = True

# Explanation schemas
class ShapValue(BaseModel):
    feature: str
    value: float
    feature_value: Any

class LimeFeature(BaseModel):
    feature: str
    weight: float

class Counterfactual(BaseModel):
    features: Dict[str, Any]
    changes: List[str]
    num_changes: int
    success: bool

class ExplanationResponse(BaseModel):
    shap_values: List[ShapValue]
    lime_explanation: List[LimeFeature]
    counterfactuals: Optional[List[Counterfactual]] = None
    
    class Config:
        from_attributes = True

# Loan terms schemas
class LoanTermsResponse(BaseModel):
    loan_amount: float
    interest_rate: float
    tenure_months: int
    monthly_emi: float
    total_payable: float
    
    class Config:
        from_attributes = True

# Full application detail
class ApplicationDetail(ApplicationResponse):
    features: Dict[str, Any]
    decision: Optional[DecisionResponse] = None
    explanation: Optional[ExplanationResponse] = None
    loan_terms: Optional[LoanTermsResponse] = None

# Officer schemas
class OfficerAction(BaseModel):
    action: str = Field(..., description="approve, reject, request_documents")
    justification: Optional[str] = None

class AuditLogResponse(BaseModel):
    id: int
    application_id: int
    officer_id: Optional[int]
    action: str
    justification: Optional[str]
    previous_status: Optional[str]
    new_status: Optional[str]
    timestamp: datetime
    
    class Config:
        from_attributes = True

# Fairness schemas
class FairnessMetricsResponse(BaseModel):
    age_dpd: Optional[float]
    children_dpd: Optional[float]
    has_alert: bool
    alert_message: Optional[str]
    metrics_detail: Dict[str, Any]
    calculated_at: datetime
    
    class Config:
        from_attributes = True

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
