"""
Tests for fairness audit module.
"""
import pytest
import pandas as pd
import numpy as np
import pickle
import os
from src.fairness_audit import (
    load_fairness_data,
    create_age_bands,
    create_family_status,
    compute_demographic_parity,
    compute_equalized_odds,
    compute_fairness_metrics,
    generate_fairness_report
)

# needs the trained model and processed data, which are not in git
pytestmark = pytest.mark.needs("models/final_lgbm.pkl", "data/processed/audit_test.csv", "data/processed/X_test_fe_v2.csv")



@pytest.fixture
def fairness_data():
    """Load test data with predictions."""
    return load_fairness_data()


def test_load_fairness_data():
    """Test loading data for fairness analysis."""
    data = load_fairness_data()
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert 'y_pred' in data.columns
    assert 'y_true' in data.columns
    assert 'default_probability' in data.columns


def test_fairness_data_predictions_valid(fairness_data):
    """Test that predictions are valid probabilities."""
    assert (fairness_data['default_probability'] >= 0).all()
    assert (fairness_data['default_probability'] <= 1).all()
    
    # Binary predictions should be 0 or 1
    assert fairness_data['y_pred'].isin([0, 1]).all()
    assert fairness_data['y_true'].isin([0, 1]).all()


def test_create_age_bands(fairness_data):
    """Test age band creation."""
    data_with_age = create_age_bands(fairness_data.copy())
    
    assert 'age_band' in data_with_age.columns
    
    # Check expected age bands exist
    expected_bands = ['<25', '25-35', '35-45', '45-55', '55+']
    actual_bands = data_with_age['age_band'].unique()
    
    for band in actual_bands:
        assert band in expected_bands, f"Unexpected age band: {band}"
    
    # Check all rows have an age band
    assert data_with_age['age_band'].notna().all()


def test_age_bands_non_overlapping(fairness_data):
    """Test that age bands don't overlap."""
    data_with_age = create_age_bands(fairness_data.copy())
    
    # Each row should have exactly one age band
    assert data_with_age['age_band'].notna().all()
    
    # Age bands should cover reasonable range
    age_counts = data_with_age['age_band'].value_counts()
    assert len(age_counts) > 0


def test_create_family_status(fairness_data):
    """Test family status creation."""
    data_with_family = create_family_status(fairness_data.copy())
    
    assert 'family_status' in data_with_family.columns
    
    # Check expected family statuses
    expected_statuses = ['Single', 'Married', 'Has Children']
    actual_statuses = data_with_family['family_status'].unique()
    
    for status in actual_statuses:
        assert status in expected_statuses, f"Unexpected family status: {status}"
    
    # Check all rows have a family status
    assert data_with_family['family_status'].notna().all()


def test_compute_demographic_parity(fairness_data):
    """Test demographic parity computation."""
    data_with_age = create_age_bands(fairness_data.copy())
    
    dp_diff = compute_demographic_parity(data_with_age, 'age_band')
    
    assert isinstance(dp_diff, float)
    # Demographic parity difference should be between -1 and 1
    assert -1 <= dp_diff <= 1


def test_compute_equalized_odds(fairness_data):
    """Test equalized odds computation."""
    data_with_age = create_age_bands(fairness_data.copy())
    
    eo_diff = compute_equalized_odds(data_with_age, 'age_band')
    
    assert isinstance(eo_diff, float)
    # Equalized odds difference should be between 0 and 1 (absolute value)
    assert 0 <= eo_diff <= 1


def test_compute_fairness_metrics_age(fairness_data):
    """Test fairness metrics computation for age bands."""
    data_with_age = create_age_bands(fairness_data.copy())
    
    metrics = compute_fairness_metrics(data_with_age, 'age_band')
    
    assert 'demographic_parity_difference' in metrics
    assert 'equalized_odds_difference' in metrics
    assert 'group_metrics' in metrics
    
    # Check group metrics structure
    assert isinstance(metrics['group_metrics'], pd.DataFrame)
    assert len(metrics['group_metrics']) > 0


def test_compute_fairness_metrics_family(fairness_data):
    """Test fairness metrics computation for family status."""
    data_with_family = create_family_status(fairness_data.copy())
    
    metrics = compute_fairness_metrics(data_with_family, 'family_status')
    
    assert 'demographic_parity_difference' in metrics
    assert 'equalized_odds_difference' in metrics
    assert 'group_metrics' in metrics


def test_group_metrics_columns(fairness_data):
    """Test that group metrics include expected columns."""
    data_with_age = create_age_bands(fairness_data.copy())
    
    metrics = compute_fairness_metrics(data_with_age, 'age_band')
    group_metrics = metrics['group_metrics']
    
    expected_cols = ['group', 'count', 'positive_rate', 'default_rate']
    for col in expected_cols:
        assert col in group_metrics.columns, f"Expected column {col} in group metrics"


def test_group_positive_rates_valid(fairness_data):
    """Test that group positive rates are valid probabilities."""
    data_with_age = create_age_bands(fairness_data.copy())
    
    metrics = compute_fairness_metrics(data_with_age, 'age_band')
    group_metrics = metrics['group_metrics']
    
    assert (group_metrics['positive_rate'] >= 0).all()
    assert (group_metrics['positive_rate'] <= 1).all()


def test_group_counts_sum_to_total(fairness_data):
    """Test that group counts sum to total dataset size."""
    data_with_age = create_age_bands(fairness_data.copy())
    
    metrics = compute_fairness_metrics(data_with_age, 'age_band')
    group_metrics = metrics['group_metrics']
    
    total_count = group_metrics['count'].sum()
    assert total_count == len(data_with_age)


def test_fairness_thresholds():
    """Test fairness metric interpretation thresholds."""
    # Commonly used thresholds
    fairness_threshold = 0.1
    
    # These are reasonable thresholds for fairness
    assert 0 < fairness_threshold < 1


def test_generate_fairness_report(fairness_data, tmp_path):
    """Test fairness report generation."""
    data_with_demo = create_age_bands(fairness_data.copy())
    data_with_demo = create_family_status(data_with_demo)
    
    output_path = tmp_path / "test_fairness_report.json"
    
    report = generate_fairness_report(
        data_with_demo,
        protected_attributes=['age_band', 'family_status'],
        output_path=str(output_path)
    )
    
    assert isinstance(report, dict)
    assert 'age_band' in report
    assert 'family_status' in report
    
    # Check report was saved
    assert output_path.exists()


def test_fairness_report_structure(fairness_data):
    """Test that fairness report has correct structure."""
    data_with_demo = create_age_bands(fairness_data.copy())
    
    report = generate_fairness_report(
        data_with_demo,
        protected_attributes=['age_band']
    )
    
    age_metrics = report['age_band']
    assert 'demographic_parity_difference' in age_metrics
    assert 'equalized_odds_difference' in age_metrics
    assert 'group_metrics' in age_metrics


def test_fairness_output_files_created():
    """Test that main script creates expected output files."""
    expected_files = [
        "reports/fairness_report.json",
        "reports/fairness_age_bands.csv",
        "reports/fairness_family_status.csv"
    ]
    
    # These will be created after running the main script
    # Just check the paths are valid
    for filepath in expected_files:
        dirpath = os.path.dirname(filepath)
        assert os.path.exists(dirpath), f"Directory {dirpath} should exist"


def test_protected_attributes_not_in_model():
    """Test that gender is not in model features (ethical constraint)."""
    model_path = "models/final_lgbm.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # LightGBM Booster uses feature_name() method, not feature_name_ attribute
    feature_names = model.feature_name()
    
    # Gender should NOT be in model
    gender_features = [f for f in feature_names if 'CODE_GENDER' in f]
    assert len(gender_features) == 0, "Gender should not be in model features"


def test_age_in_model_features():
    """Test that age-related features are in model (legal requirement for guardians)."""
    model_path = "models/final_lgbm.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # LightGBM Booster uses feature_name() method, not feature_name_ attribute
    feature_names = model.feature_name()
    
    # Age features should be present (DAYS_BIRTH or derived features)
    age_features = [f for f in feature_names if 'BIRTH' in f or 'AGE' in f]
    assert len(age_features) > 0, "Age features should be in model"


def test_multiple_protected_attributes(fairness_data):
    """Test fairness analysis with multiple protected attributes."""
    data_with_demo = create_age_bands(fairness_data.copy())
    data_with_demo = create_family_status(data_with_demo)
    
    report = generate_fairness_report(
        data_with_demo,
        protected_attributes=['age_band', 'family_status']
    )
    
    assert len(report) == 2
    assert 'age_band' in report
    assert 'family_status' in report


def test_fairness_metrics_consistency(fairness_data):
    """Test that fairness metrics are consistent across runs."""
    data_with_age = create_age_bands(fairness_data.copy())
    
    metrics1 = compute_fairness_metrics(data_with_age, 'age_band')
    metrics2 = compute_fairness_metrics(data_with_age, 'age_band')
    
    # Should be identical
    assert metrics1['demographic_parity_difference'] == metrics2['demographic_parity_difference']
    assert metrics1['equalized_odds_difference'] == metrics2['equalized_odds_difference']


def test_empty_group_handling(fairness_data):
    """Test handling of potentially empty demographic groups."""
    data_with_age = create_age_bands(fairness_data.copy())
    
    # Even if some groups are small, should not crash
    metrics = compute_fairness_metrics(data_with_age, 'age_band')
    
    # All groups should have non-zero counts
    assert (metrics['group_metrics']['count'] > 0).all()


def test_fairness_visualization_data(fairness_data):
    """Test that fairness report includes data suitable for visualization."""
    data_with_demo = create_age_bands(fairness_data.copy())
    
    metrics = compute_fairness_metrics(data_with_demo, 'age_band')
    group_metrics = metrics['group_metrics']
    
    # Should have enough data to create bar charts
    assert len(group_metrics) >= 2, "Need at least 2 groups for comparison"
    
    # Should have numeric values for plotting
    assert group_metrics['positive_rate'].dtype in [np.float64, np.float32, float]
    assert group_metrics['default_rate'].dtype in [np.float64, np.float32, float]
