from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from backend.database import get_db
from backend.models import User, FairnessMetric
from backend.schemas import FairnessMetricsResponse, DemographicStats
from backend.auth import get_current_officer
from backend.services.fairness_service import get_fairness_service

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
    fairness_service = get_fairness_service()
    stats = fairness_service.get_demographic_statistics(db, 'age')
    
    return [
        DemographicStats(
            group=s['group'],
            count=s['count'],
            approved=s['approved'],
            rejected=s['count'] - s['approved'],
            approval_rate=s['approval_rate']
        )
        for s in stats
    ]

@router.get("/demographics/family", response_model=List[DemographicStats])
def get_family_status_statistics(
    current_officer: User = Depends(get_current_officer),
    db: Session = Depends(get_db)
):
    """Get approval statistics by family status"""
    fairness_service = get_fairness_service()
    stats = fairness_service.get_demographic_statistics(db, 'family')
    
    return [
        DemographicStats(
            group=s['group'],
            count=s['count'],
            approved=s['approved'],
            rejected=s['count'] - s['approved'],
            approval_rate=s['approval_rate']
        )
        for s in stats
    ]

@router.post("/calculate")
def recalculate_fairness_metrics(
    current_officer: User = Depends(get_current_officer),
    db: Session = Depends(get_db)
):
    """Trigger recalculation of fairness metrics"""
    fairness_service = get_fairness_service()
    
    # Calculate metrics
    metrics = fairness_service.calculate_fairness_metrics(db, min_applications=50)
    
    if metrics is None:
        return {
            "message": "Insufficient data for fairness calculation (minimum 50 applications required)",
            "success": False
        }
    
    # Save to database
    fairness_record = fairness_service.save_fairness_metrics(db, metrics)
    
    return {
        "message": "Fairness metrics calculated successfully",
        "success": True,
        "metrics": {
            "age_dpd": metrics.get('age_dpd'),
            "family_dpd": metrics.get('family_dpd'),
            "has_alert": metrics.get('has_alert'),
            "total_applications": metrics.get('total_applications')
        }
    }
