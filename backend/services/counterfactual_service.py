"""
Counterfactual Service - Generate actionable recommendations
"""
import pandas as pd
import numpy as np
import dice_ml
from typing import Dict, Any, List
from backend.services.prediction_service import get_prediction_service
from backend.config import settings

# Actionable features (from Phase 3)
ACTIONABLE_FEATURES = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE',
    'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'FLAG_MOBIL',
    'FLAG_OWN_CAR_Y', 'OWN_CAR_AGE', 'CNT_FAM_MEMBERS',
    'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'INCOME_PER_PERSON',
    'CREDIT_GOODS_RATIO', 'EMPLOYMENT_YEARS', 'DOCUMENT_COUNT',
    'CONTACT_METHOD_COUNT', 'REGION_RATING_CLIENT', 'HOUR_APPR_PROCESS_START',
    'INCOME_PER_FAMILY_MEMBER', 'CREDIT_ANNUITY_RATIO'
]

class CounterfactualService:
    """Service for generating counterfactual explanations"""
    
    def __init__(self):
        self.prediction_service = get_prediction_service()
        self.dice_model = None
        self.dice_data = None
        self._initialize_dice()
    
    def _initialize_dice(self):
        """Initialize DiCE model"""
        if self.prediction_service.model is None:
            print("⚠ Warning: Cannot initialize DiCE without model")
            return
        
        try:
            # Create a wrapper for LightGBM model
            class LGBMWrapper:
                def __init__(self, model):
                    self.model = model
                
                def predict(self, X):
                    return self.model.predict(X)
                
                def predict_proba(self, X):
                    return self.model.predict_proba(X)
            
            # Wrap the model
            wrapped_model = LGBMWrapper(self.prediction_service.model)
            
            # Create DiCE ML model
            self.dice_model = dice_ml.Model(model=wrapped_model, backend='sklearn')
            
            print("✓ DiCE counterfactual generator initialized")
            
        except Exception as e:
            print(f"⚠ Warning: Could not initialize DiCE: {e}")
    
    def generate_counterfactuals(
        self, 
        features: Dict[str, Any], 
        num_counterfactuals: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate counterfactual recommendations for rejected applications
        
        Args:
            features: Original feature values
            num_counterfactuals: Number of counterfactuals to generate
            
        Returns:
            List of counterfactual recommendations
        """
        if self.dice_model is None:
            return []
        
        try:
            # Preprocess features
            X = self.prediction_service.preprocess_features(features)
            
            # Filter to actionable features only
            actionable_in_model = [f for f in ACTIONABLE_FEATURES if f in X.columns]
            
            if len(actionable_in_model) == 0:
                print("⚠ No actionable features found in model")
                return []
            
            # Create temporary DiCE data object
            dice_data = dice_ml.Data(
                dataframe=X,
                continuous_features=X.columns.tolist(),  # Treat all as continuous
                outcome_name='outcome'  # Placeholder
            )
            
            # Add outcome column (0 for rejected)
            X_with_outcome = X.copy()
            X_with_outcome['outcome'] = 0
            
            # Create DiCE explainer
            dice_exp = dice_ml.Dice(dice_data, self.dice_model)
            
            # Generate counterfactuals
            counterfactuals_result = dice_exp.generate_counterfactuals(
                query_instances=X_with_outcome.drop('outcome', axis=1),
                total_CFs=num_counterfactuals,
                desired_class="opposite",  # Flip decision
                features_to_vary=actionable_in_model
            )
            
            # Extract counterfactuals
            cf_list = []
            if hasattr(counterfactuals_result, 'cf_examples_list'):
                for cf_example in counterfactuals_result.cf_examples_list:
                    if cf_example.final_cfs_df is not None:
                        for idx in range(len(cf_example.final_cfs_df)):
                            cf_row = cf_example.final_cfs_df.iloc[idx]
                            
                            # Find changed features
                            changes = []
                            for feature in actionable_in_model:
                                if feature in X.columns:
                                    original_val = X.iloc[0][feature]
                                    new_val = cf_row[feature]
                                    
                                    if abs(original_val - new_val) > 0.001:  # Changed
                                        changes.append({
                                            "feature": feature,
                                            "original": float(original_val),
                                            "recommended": float(new_val),
                                            "change": float(new_val - original_val)
                                        })
                            
                            if changes:
                                # Calculate feasibility (fewer changes = more feasible)
                                feasibility = 1.0 / (len(changes) + 1)
                                
                                cf_list.append({
                                    "changes": changes,
                                    "num_changes": len(changes),
                                    "feasibility_score": round(feasibility, 3)
                                })
            
            # Sort by feasibility (fewer changes first)
            cf_list.sort(key=lambda x: x["num_changes"])
            
            return cf_list[:num_counterfactuals]
        
        except Exception as e:
            print(f"⚠ Counterfactual generation failed: {e}")
            import traceback
            traceback.print_exc()
            return []

# Singleton instance
_counterfactual_service = None

def get_counterfactual_service() -> CounterfactualService:
    """Get or create counterfactual service singleton"""
    global _counterfactual_service
    if _counterfactual_service is None:
        _counterfactual_service = CounterfactualService()
    return _counterfactual_service
