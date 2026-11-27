from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.orm import Session
from typing import List
from backend.database import get_db
from backend.models import User, Application, ApplicationStatus, AuditLog, DecisionType, FairnessMetric
from backend.schemas import ApplicationDetail, OfficerAction, OfficerActionResponse, AuditLogResponse, ApprovalRateStats
from backend.auth import get_current_officer
from backend.serializers import application_detail
from backend.services.loan_calculator import loan_terms_for
from backend.services.eligibility_service import GUARDIAN_DOCS, is_minor
from backend.services.report_service import compliance_report

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
    
    return [application_detail(a) for a in applications]

@router.post("/{application_id}/action", response_model=OfficerActionResponse)
def officer_action(
    application_id: int,
    action_data: OfficerAction,
    current_officer: User = Depends(get_current_officer),
    db: Session = Depends(get_db)
):
    """
    Officer action on an application: approve, reject or request_documents.
    Going against the model (approving a Review/Reject case, rejecting an
    Approve case) and requesting documents both need a written justification.
    """
    application = db.query(Application).filter(
        Application.id == application_id
    ).first()

    if not application:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found"
        )

    action = action_data.action
    if action not in ("approve", "reject", "request_documents"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid action"
        )

    decision = application.decision
    if action in ("approve", "reject") and decision is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Application has not been scored yet"
        )

    justification = (action_data.justification or "").strip()
    overrides_model = decision is not None and (
        (action == "approve" and decision.decision != DecisionType.APPROVE)
        or (action == "reject" and decision.decision == DecisionType.APPROVE)
    )
    if (overrides_model or action == "request_documents") and not justification:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A justification is required for this action"
        )

    if action == "approve" and is_minor(application.features_json):
        uploaded = {d.doc_type for d in application.documents}
        missing = [t for t in GUARDIAN_DOCS if t not in uploaded]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The guardian's documents must be uploaded before approval: " + ", ".join(missing)
            )

    previous_status = application.status
    if action == "approve":
        application.status = ApplicationStatus.APPROVED
        if decision.loan_terms is None:
            try:
                db.add(loan_terms_for(application, decision))
            except ValueError as e:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    elif action == "reject":
        application.status = ApplicationStatus.REJECTED
    else:
        application.status = ApplicationStatus.UNDER_REVIEW

    db.add(AuditLog(
        application_id=application.id,
        officer_id=current_officer.id,
        action=action,
        justification=justification or None,
        previous_status=previous_status.value,
        new_status=application.status.value
    ))
    db.commit()
    db.refresh(application)

    return {
        "message": "Action completed successfully",
        "application": application,
        "loan_terms": application.decision.loan_terms if application.decision else None,
    }

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
    if not db.query(Application).filter(Application.id == application_id).first():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found"
        )
    audit_logs = db.query(AuditLog).filter(
        AuditLog.application_id == application_id
    ).order_by(AuditLog.timestamp.desc()).all()
    
    return audit_logs


@router.get("/{application_id}/compliance-report.pdf")
def download_compliance_report(
    application_id: int,
    current_officer: User = Depends(get_current_officer),
    db: Session = Depends(get_db)
):
    """Compliance report (PDF) for one application. Generating it is recorded in the audit log."""
    application = db.query(Application).filter(Application.id == application_id).first()
    if not application:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")
    if application.decision is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Application has not been scored yet")

    logs = db.query(AuditLog).filter(AuditLog.application_id == application_id).order_by(AuditLog.timestamp.asc()).all()
    fairness = db.query(FairnessMetric).order_by(FairnessMetric.calculated_at.desc()).first()
    pdf = compliance_report(application, logs, fairness)

    db.add(AuditLog(
        application_id=application.id,
        officer_id=current_officer.id,
        action="compliance_report",
        justification="Compliance report generated",
        previous_status=application.status.value,
        new_status=application.status.value,
    ))
    db.commit()

    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="creditxplain-compliance-{application.id}.pdf"'},
    )
