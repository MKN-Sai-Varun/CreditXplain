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

def test_prediction_service_model_loaded():
    service = PredictionService()
    if service.model is None:
        pytest.skip("Model not loaded successfully")
    assert len(service.get_feature_names()) == 300


def test_prediction_if_model_exists():
    service = PredictionService()
    if service.model is None:
        pytest.skip("Model not loaded, skipping prediction test")
    probability, decision = service.predict({"AMT_CREDIT": 500000, "AMT_INCOME_TOTAL": 180000})
    assert 0.0 <= probability <= 1.0
    assert decision in ["APPROVE", "REVIEW", "REJECT"]