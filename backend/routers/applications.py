from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from backend.database import get_db
from backend.models import User, Application, ApplicationStatus
from backend.schemas import ApplicationCreate, ApplicationResponse, ApplicationDetail
from backend.auth import get_current_user

router = APIRouter()

@router.post("/", response_model=ApplicationResponse, status_code=status.HTTP_201_CREATED)
def create_application(
    application_data: ApplicationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit a new credit application"""
    new_application = Application(
        user_id=current_user.id,
        features_json=application_data.features,
        status=ApplicationStatus.SUBMITTED
    )
    
    db.add(new_application)
    db.commit()
    db.refresh(new_application)
    
    return new_application

@router.get("/", response_model=List[ApplicationResponse])
def get_user_applications(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all applications for current user"""
    applications = db.query(Application).filter(
        Application.user_id == current_user.id
    ).order_by(Application.created_at.desc()).all()
    
    return applications

@router.get("/{application_id}", response_model=ApplicationDetail)
def get_application_detail(
    application_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific application"""
    application = db.query(Application).filter(
        Application.id == application_id
    ).first()
    
    if not application:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found"
        )
    
    # Check authorization (user can only see their own applications, unless officer)
    if current_user.role == "applicant" and application.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this application"
        )
    
    # Build response with related data
    response = {
        "id": application.id,
        "user_id": application.user_id,
        "status": application.status,
        "created_at": application.created_at,
        "updated_at": application.updated_at,
        "features": application.features_json
    }
    
    # Add decision if exists
    if application.decision:
        response["decision"] = application.decision
        
        # Add explanation if exists
        if application.decision.explanation:
            response["explanation"] = application.decision.explanation
        
        # Add loan terms if exists
        if application.decision.loan_terms:
            response["loan_terms"] = application.decision.loan_terms
    
    return response
