"""
Prediction Service - ML model loading and prediction
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from backend.config import settings

class PredictionService:
    """Service for loading model and making predictions"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.continuous_cols = None
        self._load_model()
    
    def _load_model(self):
        """Load trained LightGBM model, scaler, and continuous columns"""
        try:
            # Load model
            with open(settings.MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            with open(settings.SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load continuous columns list
            with open(settings.CONTINUOUS_COLS_PATH, 'rb') as f:
                self.continuous_cols = pickle.load(f)
            
            print(f"✓ Model loaded from {settings.MODEL_PATH}")
            print(f"✓ Scaler loaded from {settings.SCALER_PATH}")
            print(f"✓ Continuous columns loaded: {len(self.continuous_cols)} features")
            
        except Exception as e:
            print(f"⚠ Warning: Could not load model files: {e}")
            print("  Prediction service will not work until models are available")
            self.model = None
            self.scaler = None
            self.continuous_cols = []
    
    def preprocess_features(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess raw features for prediction
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Preprocessed DataFrame ready for model
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Ensure all continuous columns exist (fill missing with 0)
        for col in self.continuous_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Select only continuous columns (match training data)
        df = df[self.continuous_cols]
        
        # Apply scaling
        df_scaled = pd.DataFrame(
            self.scaler.transform(df),
            columns=self.continuous_cols,
            index=df.index
        )
        
        return df_scaled
    
    def predict(self, features: Dict[str, Any]) -> Tuple[float, str]:
        """
        Make prediction on preprocessed features
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Tuple of (probability, decision)
            - probability: Default probability (0-1)
            - decision: APPROVE / REVIEW / REJECT
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Preprocess features
        X = self.preprocess_features(features)
        
        # Get prediction probability
        probability = self.model.predict_proba(X)[0][1]  # Probability of default (class 1)
        
        # Apply decision router
        if probability < settings.APPROVE_THRESHOLD:
            decision = "APPROVE"
        elif probability < settings.REJECT_THRESHOLD:
            decision = "REVIEW"
        else:
            decision = "REJECT"
        
        return float(probability), decision
    
    def get_feature_names(self):
        """Get list of feature names expected by model"""
        if self.model is None:
            return []
        return self.continuous_cols

# Singleton instance
_prediction_service = None

def get_prediction_service() -> PredictionService:
    """Get or create prediction service singleton"""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service
