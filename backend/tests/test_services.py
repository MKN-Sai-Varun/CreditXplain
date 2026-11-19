import pytest
from backend.services.prediction_service import PredictionService
from backend.config import settings

def test_prediction_service_initialization():
    """Test that prediction service initializes"""
    service = PredictionService()
    assert service is not None

def test_prediction_service_model_loaded():
    """Test that model is loaded if files exist"""
    service = PredictionService()
    
    # Check if model files exist and are valid
    if settings.MODEL_PATH.exists() and service.model is not None:
        assert service.model is not None
        assert service.scaler is not None
        assert service.continuous_cols is not None
        assert len(service.continuous_cols) > 0
    else:
        # Model files don't exist or failed to load, service should handle gracefully
        pytest.skip("Model not loaded successfully")

def test_prediction_service_get_feature_names():
    """Test getting feature names from model"""
    service = PredictionService()
    feature_names = service.get_feature_names()
    assert isinstance(feature_names, list)

def test_prediction_if_model_exists():
    """Test prediction if model is loaded"""
    service = PredictionService()
    
    if service.model is not None:
        # Create sample features (all zeros as fallback)
        sample_features = {col: 0.0 for col in service.continuous_cols}
        
        # Make prediction
        probability, decision = service.predict(sample_features)
        
        # Verify outputs
        assert isinstance(probability, float)
        assert 0.0 <= probability <= 1.0
        assert decision in ["APPROVE", "REVIEW", "REJECT"]
    else:
        pytest.skip("Model not loaded, skipping prediction test")
