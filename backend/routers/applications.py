from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.orm import Session
from typing import List
from backend.database import get_db
from backend.models import User, Application, ApplicationStatus
from backend.schemas import ApplicationCreate, ApplicationResponse, ApplicationDetail
from backend.auth import get_current_user
from backend.serializers import application_detail
from backend.services.report_service import applicant_report

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
        tenure_months=application_data.tenure_months,
        recourse_preferences=(application_data.recourse_preferences.model_dump()
                              if application_data.recourse_preferences else None),
        guardian=application_data.guardian.model_dump() if application_data.guardian else None,
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
    
    return application_detail(application)


@router.get("/{application_id}/report.pdf")
def download_decision_report(
    application_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Decision report (PDF): decision, main reasons, recourse options, loan terms, document requests"""
    application = db.query(Application).filter(Application.id == application_id).first()
    if not application:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")
    if current_user.role == "applicant" and application.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this application")
    if application.decision is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Application has not been scored yet")

    return Response(
        content=applicant_report(application),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="creditxplain-application-{application.id}.pdf"'},
    )
