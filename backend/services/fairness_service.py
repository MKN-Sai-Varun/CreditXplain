"""
Fairness Monitoring Service - Calculate demographic parity metrics
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sqlalchemy.orm import Session
from backend.models import Application, Decision, FairnessMetric, DecisionType

class FairnessService:
    """Service for calculating fairness metrics"""
    
    def __init__(self):
        self.fairness_threshold = 0.1  # Demographic parity threshold
    
    def create_age_bands(self, days_birth_series: pd.Series) -> pd.Series:
        """
        Create age bands from DAYS_BIRTH
        
        Args:
            days_birth_series: Series of DAYS_BIRTH values (negative days)
            
        Returns:
            Series of age band labels
        """
        # Convert DAYS_BIRTH to years
        age_years = (-days_birth_series / 365).astype(int)
        
        # Create bands
        age_bands = pd.cut(
            age_years,
            bins=[0, 25, 35, 45, 55, 100],
            labels=['<25', '25-35', '35-45', '45-55', '55+'],
            right=False
        )
        
        return age_bands
    
    def create_family_status(self, df: pd.DataFrame) -> pd.Series:
        """
        Create family status from children count
        
        Args:
            df: DataFrame with CNT_CHILDREN or similar columns
            
        Returns:
            Series of family status (Has Children / No Children)
        """
        if 'CNT_CHILDREN' in df.columns:
            has_children = df['CNT_CHILDREN'] > 0
        elif 'has_children' in df.columns:
            has_children = df['has_children']
        else:
            # Default to No Children if column missing
            has_children = pd.Series(False, index=df.index)
        
        return has_children.map({True: 'Has Children', False: 'No Children'})
    
    def calculate_fairness_metrics(
        self,
        db: Session,
        min_applications: int = 50
    ) -> Optional[Dict]:
        """
        Calculate fairness metrics from database
        
        Args:
            db: Database session
            min_applications: Minimum applications required for calculation
            
        Returns:
            Dictionary with fairness metrics or None if insufficient data
        """
        # Get all applications with decisions
        applications = db.query(Application, Decision).join(
            Decision, Application.id == Decision.application_id
        ).all()
        
        if len(applications) < min_applications:
            return None
        
        # Build DataFrame
        data = []
        for app, decision in applications:
            features = app.features_json
            
            data.append({
                'application_id': app.id,
                'decision': decision.decision.value,
                'probability': decision.probability,
                'approved': 1 if decision.decision == DecisionType.APPROVE else 0,
                'DAYS_BIRTH': features.get('DAYS_BIRTH', 0),
                'CNT_CHILDREN': features.get('CNT_CHILDREN', 0)
            })
        
        df = pd.DataFrame(data)
        
        # Create demographic features
        df['age_band'] = self.create_age_bands(df['DAYS_BIRTH'])
        df['family_status'] = self.create_family_status(df)
        
        # Calculate metrics
        metrics = {}
        
        # Age-based metrics
        if df['age_band'].notna().any():
            age_dpd = demographic_parity_difference(
                y_true=df['approved'],
                y_pred=df['approved'],
                sensitive_features=df['age_band']
            )
            
            age_eod = equalized_odds_difference(
                y_true=df['approved'],
                y_pred=df['approved'],
                sensitive_features=df['age_band']
            )
            
            metrics['age_dpd'] = float(age_dpd)
            metrics['age_eod'] = float(age_eod)
            
            # Calculate approval rates by age band
            age_stats = df.groupby('age_band').agg({
                'approved': ['sum', 'count', 'mean']
            }).round(4)
            
            metrics['age_band_stats'] = age_stats.to_dict()
        
        # Family status metrics
        if df['family_status'].notna().any():
            family_dpd = demographic_parity_difference(
                y_true=df['approved'],
                y_pred=df['approved'],
                sensitive_features=df['family_status']
            )
            
            family_eod = equalized_odds_difference(
                y_true=df['approved'],
                y_pred=df['approved'],
                sensitive_features=df['family_status']
            )
            
            metrics['family_dpd'] = float(family_dpd)
            metrics['family_eod'] = float(family_eod)
            
            # Calculate approval rates by family status
            family_stats = df.groupby('family_status').agg({
                'approved': ['sum', 'count', 'mean']
            }).round(4)
            
            metrics['family_status_stats'] = family_stats.to_dict()
        
        # Check for alerts
        has_alert = False
        alert_messages = []
        
        if metrics.get('age_dpd', 0) > self.fairness_threshold:
            has_alert = True
            alert_messages.append(
                f"Age-based disparity ({metrics['age_dpd']:.4f}) exceeds threshold ({self.fairness_threshold})"
            )
        
        if metrics.get('family_dpd', 0) > self.fairness_threshold:
            has_alert = True
            alert_messages.append(
                f"Family status disparity ({metrics['family_dpd']:.4f}) exceeds threshold ({self.fairness_threshold})"
            )
        
        metrics['has_alert'] = has_alert
        metrics['alert_messages'] = alert_messages
        metrics['total_applications'] = len(df)
        metrics['calculated_at'] = datetime.utcnow().isoformat()
        
        return metrics
    
    def save_fairness_metrics(self, db: Session, metrics: Dict) -> FairnessMetric:
        """
        Save fairness metrics to database
        
        Args:
            db: Database session
            metrics: Metrics dictionary from calculate_fairness_metrics
            
        Returns:
            Saved FairnessMetric record
        """
        fairness_record = FairnessMetric(
            age_dpd=metrics.get('age_dpd'),
            children_dpd=metrics.get('family_dpd'),
            metrics_json=metrics,
            has_alert=1 if metrics.get('has_alert') else 0,
            alert_message='; '.join(metrics.get('alert_messages', []))
        )
        
        db.add(fairness_record)
        db.commit()
        db.refresh(fairness_record)
        
        return fairness_record
    
    def get_demographic_statistics(
        self,
        db: Session,
        demographic: str
    ) -> List[Dict]:
        """
        Get approval statistics by demographic group
        
        Args:
            db: Database session
            demographic: 'age' or 'family'
            
        Returns:
            List of dictionaries with group statistics
        """
        # Get latest fairness metrics
        latest_metric = db.query(FairnessMetric).order_by(
            FairnessMetric.calculated_at.desc()
        ).first()
        
        if not latest_metric or not latest_metric.metrics_json:
            return []
        
        metrics = latest_metric.metrics_json
        
        if demographic == 'age' and 'age_band_stats' in metrics:
            stats_dict = metrics['age_band_stats']
            result = []
            
            for group in stats_dict.get('approved', {}).get('sum', {}).keys():
                result.append({
                    'group': group,
                    'count': int(stats_dict['approved']['count'].get(group, 0)),
                    'approved': int(stats_dict['approved']['sum'].get(group, 0)),
                    'approval_rate': float(stats_dict['approved']['mean'].get(group, 0))
                })
            
            return result
        
        elif demographic == 'family' and 'family_status_stats' in metrics:
            stats_dict = metrics['family_status_stats']
            result = []
            
            for group in stats_dict.get('approved', {}).get('sum', {}).keys():
                result.append({
                    'group': group,
                    'count': int(stats_dict['approved']['count'].get(group, 0)),
                    'approved': int(stats_dict['approved']['sum'].get(group, 0)),
                    'approval_rate': float(stats_dict['approved']['mean'].get(group, 0))
                })
            
            return result
        
        return []

# Singleton instance
_fairness_service = None

def get_fairness_service() -> FairnessService:
    """Get or create fairness service singleton"""
    global _fairness_service
    if _fairness_service is None:
        _fairness_service = FairnessService()
    return _fairness_service
