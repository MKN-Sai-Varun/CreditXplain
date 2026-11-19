from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from backend.database import get_db
from backend.models import User, FairnessMetric
from backend.schemas import FairnessMetricsResponse, DemographicStats
from backend.auth import get_current_officer

router = APIRouter()

@router.get("/metrics", response_model=FairnessMetricsResponse)
def get_latest_fairness_metrics(
    current_officer: User = Depends(get_current_officer),
    db: Session = Depends(get_db)
):
    """Get the latest fairness metrics"""
    latest_metric = db.query(FairnessMetric).order_by(
        FairnessMetric.calculated_at.desc()
    ).first()
    
    if not latest_metric:
        return {
            "age_dpd": None,
            "children_dpd": None,
            "has_alert": False,
            "alert_message": None,
            "metrics_detail": {},
            "calculated_at": None
        }
    
    return {
        "age_dpd": latest_metric.age_dpd,
        "children_dpd": latest_metric.children_dpd,
        "has_alert": bool(latest_metric.has_alert),
        "alert_message": latest_metric.alert_message,
        "metrics_detail": latest_metric.metrics_json or {},
        "calculated_at": latest_metric.calculated_at
    }

@router.get("/demographics/age", response_model=List[DemographicStats])
def get_age_band_statistics(
    current_officer: User = Depends(get_current_officer),
    db: Session = Depends(get_db)
):
    """Get approval statistics by age band"""
    # TODO: Implement demographic breakdown (Phase 4.5)
    # This will require joining applications with decisions and calculating stats per age group
    return []

@router.get("/demographics/family", response_model=List[DemographicStats])
def get_family_status_statistics(
    current_officer: User = Depends(get_current_officer),
    db: Session = Depends(get_db)
):
    """Get approval statistics by family status"""
    # TODO: Implement demographic breakdown (Phase 4.5)
    return []

@router.post("/calculate")
def recalculate_fairness_metrics(
    current_officer: User = Depends(get_current_officer),
    db: Session = Depends(get_db)
):
    """Trigger recalculation of fairness metrics"""
    # TODO: Implement fairness calculation (Phase 4.5)
    # This will use the fairness_audit.py logic from Phase 3
    return {"message": "Fairness metrics calculation triggered"}
