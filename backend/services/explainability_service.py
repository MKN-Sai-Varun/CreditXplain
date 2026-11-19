"""
Explainability Service - SHAP and LIME explanations
"""
import pandas as pd
import numpy as np
import shap
from lime import lime_tabular
from typing import Dict, Any, List
from backend.services.prediction_service import get_prediction_service

class ExplainabilityService:
    """Service for generating SHAP and LIME explanations"""
    
    def __init__(self):
        self.prediction_service = get_prediction_service()
        self.shap_explainer = None
        self.lime_explainer = None
        self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Initialize SHAP and LIME explainers"""
        if self.prediction_service.model is None:
            print("⚠ Warning: Cannot initialize explainers without model")
            return
        
        try:
            # Initialize SHAP TreeExplainer
            self.shap_explainer = shap.TreeExplainer(self.prediction_service.model)
            print("✓ SHAP TreeExplainer initialized")
            
            # Load sample data for LIME (use first 1000 rows of training data if available)
            try:
                train_path = self.prediction_service.settings.DATA_DIR / "X_train_fe_v2.csv"
                if train_path.exists():
                    train_sample = pd.read_csv(train_path, nrows=1000)
                    train_sample = train_sample[self.prediction_service.continuous_cols]
                    
                    # Initialize LIME explainer
                    self.lime_explainer = lime_tabular.LimeTabularExplainer(
                        training_data=train_sample.values,
                        feature_names=self.prediction_service.continuous_cols,
                        mode='classification',
                        discretize_continuous=False
                    )
                    print("✓ LIME explainer initialized with 1000 training samples")
                else:
                    print("⚠ Warning: Training data not found, LIME explainer not initialized")
            except Exception as e:
                print(f"⚠ Warning: Could not initialize LIME: {e}")
        
        except Exception as e:
            print(f"⚠ Warning: Could not initialize explainers: {e}")
    
    def generate_shap_explanation(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate SHAP explanation for a single prediction
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            List of dicts with feature, value, and SHAP value
        """
        if self.shap_explainer is None:
            return []
        
        # Preprocess features
        X = self.prediction_service.preprocess_features(features)
        
        # Get SHAP values
        shap_values = self.shap_explainer.shap_values(X)
        
        # Handle binary classification (shap_values might be list of two arrays)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class (default)
        
        # Get base value (expected value)
        base_value = self.shap_explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]
        
        # Create explanation list
        explanations = []
        for i, (feature, shap_val) in enumerate(zip(self.prediction_service.continuous_cols, shap_values[0])):
            explanations.append({
                "feature": feature,
                "value": float(X.iloc[0, i]),
                "shap_value": float(shap_val)
            })
        
        # Sort by absolute SHAP value (most important first)
        explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        
        return explanations[:20]  # Return top 20 features
    
    def generate_lime_explanation(self, features: Dict[str, Any], num_features: int = 10) -> List[Dict[str, Any]]:
        """
        Generate LIME explanation for a single prediction
        
        Args:
            features: Dictionary of feature values
            num_features: Number of top features to explain
            
        Returns:
            List of dicts with feature and weight
        """
        if self.lime_explainer is None:
            return []
        
        try:
            # Preprocess features
            X = self.prediction_service.preprocess_features(features)
            
            # Get prediction function
            def predict_fn(X_array):
                return self.prediction_service.model.predict_proba(X_array)
            
            # Generate LIME explanation
            explanation = self.lime_explainer.explain_instance(
                data_row=X.iloc[0].values,
                predict_fn=predict_fn,
                num_features=num_features
            )
            
            # Extract feature weights
            lime_explanations = []
            for feature_idx, weight in explanation.as_list():
                # Parse feature name from LIME format (e.g., "feature_name > 0.5")
                feature_name = feature_idx.split()[0] if isinstance(feature_idx, str) else self.prediction_service.continuous_cols[feature_idx]
                
                lime_explanations.append({
                    "feature": feature_name,
                    "weight": float(weight)
                })
            
            return lime_explanations
        
        except Exception as e:
            print(f"⚠ LIME explanation failed: {e}")
            return []

# Singleton instance
_explainability_service = None

def get_explainability_service() -> ExplainabilityService:
    """Get or create explainability service singleton"""
    global _explainability_service
    if _explainability_service is None:
        _explainability_service = ExplainabilityService()
    return _explainability_service
