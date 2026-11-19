"""
Tests for explainability.py
Requires explainability script to be run first: python3 src/explainability.py
"""
import pytest
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import shap


@pytest.fixture
def model():
    model_path = Path("models/final_lgbm.pkl")
    if not model_path.exists():
        pytest.skip("Model not trained. Run: python3 src/final_model.py")
    return joblib.load(model_path)


@pytest.fixture
def model_card():
    card_path = Path("models/final_model_card.json")
    if not card_path.exists():
        pytest.skip("Model card missing. Run: python3 src/final_model.py")
    
    with open(card_path) as f:
        return json.load(f)


@pytest.fixture
def test_data(model_card):
    X_test = pd.read_csv("data/processed/X_test_fe_v2.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    
    # Apply exclusions
    for feat in model_card["features_excluded"]:
        if feat in X_test.columns:
            X_test = X_test.drop(columns=[feat])
    
    gender_cols = [c for c in X_test.columns if "GENDER" in c.upper()]
    if gender_cols:
        X_test = X_test.drop(columns=gender_cols)
    
    X_test = X_test[model_card["features_used"]]
    
    return X_test, y_test


def test_shap_global_importance_exists():
    """Verify global SHAP importance file was created."""
    assert Path("reports/shap/global_feature_importance.csv").exists()


def test_shap_importance_has_all_features(model_card):
    """Verify SHAP importance includes all model features."""
    if not Path("reports/shap/global_feature_importance.csv").exists():
        pytest.skip("Run: python3 src/explainability.py")
    
    importance = pd.read_csv("reports/shap/global_feature_importance.csv")
    assert len(importance) == len(model_card["features_used"])


def test_shap_importance_nonnegative():
    """Verify SHAP importance values are non-negative."""
    if not Path("reports/shap/global_feature_importance.csv").exists():
        pytest.skip("Run: python3 src/explainability.py")
    
    importance = pd.read_csv("reports/shap/global_feature_importance.csv")
    assert (importance["shap_importance"] >= 0).all()


def test_shap_importance_sorted():
    """Verify SHAP importance is sorted descending."""
    if not Path("reports/shap/global_feature_importance.csv").exists():
        pytest.skip("Run: python3 src/explainability.py")
    
    importance = pd.read_csv("reports/shap/global_feature_importance.csv")
    assert importance["shap_importance"].is_monotonic_decreasing


def test_shap_additivity_property(model, test_data):
    """Test that SHAP values sum to prediction (additivity property)."""
    X_test, _ = test_data
    
    # Sample small subset for fast test
    X_sample = X_test.head(50)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    # SHAP sum should equal raw model prediction (log-odds, not probability)
    base_value = shap_values.base_values[0]
    shap_predictions_raw = base_value + shap_values.values.sum(axis=1)
    model_predictions_raw = model.predict(X_sample, raw_score=True)
    
    max_diff = np.abs(shap_predictions_raw - model_predictions_raw).max()
    
    # Allow small numerical error
    assert max_diff < 1e-6, f"SHAP additivity violated: max diff {max_diff}"


def test_shap_stability(model, test_data):
    """Test that SHAP values are stable across repeated runs."""
    X_test, _ = test_data
    
    # Sample small subset for fast test
    X_sample = X_test.head(20)
    
    explainer = shap.TreeExplainer(model)
    
    # Run twice
    shap_run1 = explainer(X_sample).values
    shap_run2 = explainer(X_sample).values
    
    # Should be identical (deterministic)
    correlations = []
    for i in range(len(X_sample)):
        corr = np.corrcoef(shap_run1[i], shap_run2[i])[0, 1]
        correlations.append(corr)
    
    avg_corr = np.mean(correlations)
    
    assert avg_corr > 0.999, f"SHAP is not stable: avg correlation {avg_corr:.6f}"


def test_shap_plots_generated():
    """Verify SHAP plots were generated."""
    assert Path("reports/shap/summary_beeswarm.png").exists()
    assert Path("reports/shap/feature_importance_bar.png").exists()


def test_waterfall_plots_exist():
    """Verify at least one waterfall plot was generated."""
    shap_dir = Path("reports/shap")
    waterfall_plots = list(shap_dir.glob("waterfall_applicant_*.png"))
    assert len(waterfall_plots) > 0, "No waterfall plots found"


def test_lime_explanations_exist():
    """Verify LIME explanations were generated."""
    lime_dir = Path("reports/lime")
    lime_html = list(lime_dir.glob("explanation_applicant_*.html"))
    lime_png = list(lime_dir.glob("explanation_applicant_*.png"))
    
    assert len(lime_html) > 0, "No LIME HTML explanations found"
    assert len(lime_png) > 0, "No LIME PNG explanations found"


def test_consistency_report_exists():
    """Verify SHAP-LIME consistency report was created."""
    assert Path("reports/shap_lime_consistency.csv").exists()


def test_consistency_metrics_valid():
    """Verify consistency metrics are in valid range."""
    if not Path("reports/shap_lime_consistency.csv").exists():
        pytest.skip("Run: python3 src/explainability.py")
    
    consistency = pd.read_csv("reports/shap_lime_consistency.csv")
    
    # Overlap should be between 0 and 10
    assert (consistency["top10_overlap"] >= 0).all()
    assert (consistency["top10_overlap"] <= 10).all()
    
    # Percentage should be 0-100
    assert (consistency["overlap_percentage"] >= 0).all()
    assert (consistency["overlap_percentage"] <= 100).all()


def test_explainability_report_exists():
    """Verify explainability report was generated."""
    assert Path("reports/explainability_report.md").exists()


def test_shap_values_reasonable_range(model, test_data):
    """Test that SHAP values are in a reasonable range."""
    X_test, _ = test_data
    X_sample = X_test.head(50)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    # SHAP values for probabilities should typically be between -1 and 1
    max_abs_shap = np.abs(shap_values.values).max()
    
    assert max_abs_shap < 2.0, f"SHAP values unexpectedly large: {max_abs_shap}"
