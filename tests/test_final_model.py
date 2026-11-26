"""
Tests for the final production model.
Requires the model to be trained first: python3 src/final_model.py
"""
import pytest
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def model_card():
    card_path = Path("models/final_model_card.json")
    if not card_path.exists():
        pytest.skip("Model not trained yet. Run: python3 src/final_model.py")
    
    with open(card_path) as f:
        return json.load(f)


@pytest.fixture
def model():
    model_path = Path("models/final_lgbm.pkl")
    if not model_path.exists():
        pytest.skip("Model not trained yet. Run: python3 src/final_model.py")
    
    return joblib.load(model_path)


@pytest.mark.needs("models/final_lgbm.pkl")
def test_model_exists():
    """Verify the model file was created."""
    assert Path("models/final_lgbm.pkl").exists()


def test_model_card_exists():
    """Verify the model card was created."""
    assert Path("models/final_model_card.json").exists()


def test_gender_excluded(model_card):
    """Verify gender features are not in the model."""
    features = model_card["features_used"]
    gender_features = [f for f in features if "GENDER" in f.upper()]
    assert len(gender_features) == 0, f"Found gender features: {gender_features}"


def test_days_birth_excluded(model_card):
    """Verify DAYS_BIRTH was excluded (AGE_YEARS used instead)."""
    features = model_card["features_used"]
    assert "DAYS_BIRTH" not in features


def test_model_card_has_required_fields(model_card):
    """Verify model card contains all required metadata."""
    required_fields = [
        "model_type",
        "features_used",
        "features_excluded",
        "test_roc_auc",
        "approve_threshold",
        "reject_threshold"
    ]
    
    for field in required_fields:
        assert field in model_card, f"Missing field: {field}"


def test_thresholds_ordered(model_card):
    """Verify approve < reject thresholds."""
    approve = model_card["approve_threshold"]
    reject = model_card["reject_threshold"]
    
    assert 0 <= approve < reject <= 1, \
        f"Invalid thresholds: approve={approve}, reject={reject}"


@pytest.mark.needs("data/processed/X_test_fe_v2.csv")
def test_model_produces_probabilities(model):
    """Verify model outputs are valid probabilities."""
    X_test = pd.read_csv("data/processed/X_test_fe_v2.csv")
    
    # Match feature exclusions from training
    X_test = X_test.drop(columns=["DAYS_BIRTH"], errors="ignore")
    gender_cols = [c for c in X_test.columns if "GENDER" in c.upper()]
    X_test = X_test.drop(columns=gender_cols, errors="ignore")
    
    preds = model.predict(X_test.head(100))
    
    assert np.all(preds >= 0) and np.all(preds <= 1), "Predictions outside [0, 1]"


def test_test_auc_meets_minimum(model_card):
    """Verify test AUC meets minimum acceptable threshold."""
    MIN_ACCEPTABLE_AUC = 0.70
    test_auc = model_card["test_roc_auc"]
    
    assert test_auc >= MIN_ACCEPTABLE_AUC, \
        f"Test AUC {test_auc:.4f} below minimum {MIN_ACCEPTABLE_AUC}"


def test_bands_file_created():
    """Verify decision band analysis was saved."""
    assert Path("reports/final_model_bands.csv").exists()


@pytest.mark.needs("data/processed/audit_test.csv")
def test_audit_data_created():
    """Verify audit dataset was created for Phase 3."""
    audit_path = Path("data/processed/audit_test.csv")
    assert audit_path.exists()
    
    audit = pd.read_csv(audit_path)
    required_cols = ["predicted_probability", "decision", "actual_default"]
    
    for col in required_cols:
        assert col in audit.columns, f"Missing column: {col}"


@pytest.mark.needs("data/processed/X_train_fe_v2.csv", "data/processed/y_train.csv")
def test_cutoff_rule_reproduces_model_card(model, model_card):
    """decision.select_cutoffs on the final model's validation split gives the card's cutoffs."""
    from sklearn.model_selection import train_test_split
    from decision import select_cutoffs

    X = pd.read_csv("data/processed/X_train_fe_v2.csv")[model_card["features_used"]]
    y = pd.read_csv("data/processed/y_train.csv").squeeze()
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    approve_below, reject_from = select_cutoffs(model.predict(X_val), y_val)
    assert approve_below == pytest.approx(model_card["approve_threshold"], rel=1e-9)
    assert reject_from == pytest.approx(model_card["reject_threshold"], rel=1e-9)

