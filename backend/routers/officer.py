from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from backend.database import get_db
from backend.models import User, Application, ApplicationStatus, AuditLog
from backend.schemas import ApplicationDetail, OfficerAction, AuditLogResponse, ApprovalRateStats
from backend.auth import get_current_officer

router = APIRouter()

@router.get("/queue", response_model=List[ApplicationDetail])
def get_review_queue(
    current_officer: User = Depends(get_current_officer),
    db: Session = Depends(get_db)
):
    """Get all applications pending officer review"""
    applications = db.query(Application).filter(
        Application.status == ApplicationStatus.UNDER_REVIEW
    ).order_by(Application.created_at.asc()).all()
    
    # TODO: Format with decision and explanation data
    return applications

@router.post("/{application_id}/action")
def officer_action(
    application_id: int,
    action_data: OfficerAction,
    current_officer: User = Depends(get_current_officer),
    db: Session = Depends(get_db)
):
    """Perform officer action on an application (approve/reject/request documents)"""
    application = db.query(Application).filter(
        Application.id == application_id
    ).first()
    
    if not application:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found"
        )
    
    previous_status = application.status
    
    # Update status based on action
    if action_data.action == "approve":
        application.status = ApplicationStatus.APPROVED
    elif action_data.action == "reject":
        application.status = ApplicationStatus.REJECTED
    elif action_data.action == "request_documents":
        application.status = ApplicationStatus.UNDER_REVIEW
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid action"
        )
    
    # Create audit log
    audit_log = AuditLog(
        application_id=application.id,
        officer_id=current_officer.id,
        action=action_data.action,
        justification=action_data.justification,
        previous_status=previous_status.value,
        new_status=application.status.value
    )
    
    db.add(audit_log)
    db.commit()
    db.refresh(application)
    
    return {"message": "Action completed successfully", "application": application}

@router.get("/analytics/approval-rates", response_model=ApprovalRateStats)
def get_approval_rates(
    current_officer: User = Depends(get_current_officer),
    db: Session = Depends(get_db)
):
    """Get overall approval rate statistics"""
    total = db.query(Application).count()
    approved = db.query(Application).filter(
        Application.status == ApplicationStatus.APPROVED
    ).count()
    rejected = db.query(Application).filter(
        Application.status == ApplicationStatus.REJECTED
    ).count()
    under_review = db.query(Application).filter(
        Application.status == ApplicationStatus.UNDER_REVIEW
    ).count()
    
    approval_rate = approved / total if total > 0 else 0.0
    
    return {
        "total_applications": total,
        "approved": approved,
        "rejected": rejected,
        "under_review": under_review,
        "approval_rate": approval_rate
    }

@router.get("/{application_id}/audit-log", response_model=List[AuditLogResponse])
def get_application_audit_log(
    application_id: int,
    current_officer: User = Depends(get_current_officer),
    db: Session = Depends(get_db)
):
    """Get audit log for a specific application"""
    audit_logs = db.query(AuditLog).filter(
        AuditLog.application_id == application_id
    ).order_by(AuditLog.timestamp.desc()).all()
    
    return audit_logs
