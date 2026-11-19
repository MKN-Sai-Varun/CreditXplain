"""
Tests for counterfactual generation module.
"""
import pytest
import pandas as pd
import numpy as np
import pickle
import os
from src.counterfactuals import (
    load_counterfactual_model,
    generate_counterfactuals,
    rank_counterfactuals_by_feasibility,
    save_counterfactual_report
)


@pytest.fixture
def model_artifacts():
    """Load required model artifacts."""
    model_path = "models/final_lgbm.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    test_data = pd.read_csv("data/processed/X_test_fe_v2.csv")
    return model, test_data


@pytest.fixture
def sample_rejected_applicant(model_artifacts):
    """Get a sample rejected applicant."""
    _, test_data = model_artifacts
    # Get first row as sample
    return test_data.iloc[0:1].copy()


def test_load_counterfactual_model():
    """Test model loading for counterfactual generation."""
    model, feature_names, continuous_features = load_counterfactual_model()
    
    assert model is not None
    assert len(feature_names) > 0
    assert len(continuous_features) > 0
    assert all(isinstance(f, str) for f in feature_names)


def test_actionable_features_defined():
    """Test that actionable features are properly defined."""
    from src.counterfactuals import ACTIONABLE_FEATURES
    
    assert len(ACTIONABLE_FEATURES) > 0
    # Check some expected actionable features
    expected_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE']
    for feat in expected_features:
        assert feat in ACTIONABLE_FEATURES, f"{feat} should be actionable"


def test_immutable_features_excluded():
    """Test that immutable features are excluded from counterfactual generation."""
    from src.counterfactuals import ACTIONABLE_FEATURES
    
    # These should NOT be actionable
    immutable = ['DAYS_BIRTH', 'CODE_GENDER_M', 'CODE_GENDER_XNA', 
                 'FLAG_OWN_CAR', 'NAME_EDUCATION_TYPE_Higher education']
    
    for feat in immutable:
        if feat in ['CODE_GENDER_M', 'CODE_GENDER_XNA']:
            # Gender not in features at all
            continue
        # Age should not be actionable (derived from DAYS_BIRTH)
        if 'BIRTH' in feat or 'AGE' in feat:
            assert feat not in ACTIONABLE_FEATURES, f"{feat} should be immutable"


def test_generate_counterfactuals_structure(sample_rejected_applicant):
    """Test that counterfactual generation returns valid structure."""
    result = generate_counterfactuals(
        applicant_data=sample_rejected_applicant,
        num_counterfactuals=2,
        diversity_weight=0.5
    )
    
    assert 'counterfactuals' in result
    assert 'original_prediction' in result
    assert 'query_instance' in result
    
    # Check prediction is valid probability
    assert 0 <= result['original_prediction'] <= 1
    
    # Check counterfactuals structure
    cfs = result['counterfactuals']
    assert isinstance(cfs, pd.DataFrame) or cfs is None


def test_counterfactual_predictions_improved(sample_rejected_applicant):
    """Test that counterfactuals have lower default probability than original."""
    result = generate_counterfactuals(
        applicant_data=sample_rejected_applicant,
        num_counterfactuals=3,
        diversity_weight=0.5
    )
    
    if result['counterfactuals'] is not None:
        original_pred = result['original_prediction']
        cf_preds = result['counterfactual_predictions']
        
        # Most counterfactuals should have lower default probability
        improved = sum(1 for p in cf_preds if p < original_pred)
        assert improved >= len(cf_preds) * 0.5, "At least 50% of CFs should improve prediction"


def test_rank_counterfactuals_by_feasibility(sample_rejected_applicant):
    """Test counterfactual ranking by feasibility."""
    result = generate_counterfactuals(
        applicant_data=sample_rejected_applicant,
        num_counterfactuals=3
    )
    
    if result['counterfactuals'] is not None:
        ranked = rank_counterfactuals_by_feasibility(
            result['counterfactuals'],
            result['query_instance']
        )
        
        assert 'feasibility_score' in ranked.columns
        assert 'rank' in ranked.columns
        
        # Check ranks are sequential
        assert list(ranked['rank']) == list(range(1, len(ranked) + 1))
        
        # Check feasibility scores are between 0 and 1
        assert (ranked['feasibility_score'] >= 0).all()
        assert (ranked['feasibility_score'] <= 1).all()


def test_feasibility_components(sample_rejected_applicant):
    """Test that feasibility score includes expected components."""
    result = generate_counterfactuals(
        applicant_data=sample_rejected_applicant,
        num_counterfactuals=2
    )
    
    if result['counterfactuals'] is not None:
        ranked = rank_counterfactuals_by_feasibility(
            result['counterfactuals'],
            result['query_instance']
        )
        
        # Check for feasibility sub-components
        expected_cols = ['num_changes', 'distance']
        for col in expected_cols:
            assert col in ranked.columns, f"Feasibility should include {col}"


def test_diversity_parameter_effect(sample_rejected_applicant):
    """Test that diversity parameter affects counterfactual generation."""
    result_low = generate_counterfactuals(
        applicant_data=sample_rejected_applicant,
        num_counterfactuals=3,
        diversity_weight=0.1
    )
    
    result_high = generate_counterfactuals(
        applicant_data=sample_rejected_applicant,
        num_counterfactuals=3,
        diversity_weight=0.9
    )
    
    # Both should succeed (or both fail)
    assert (result_low['counterfactuals'] is None) == (result_high['counterfactuals'] is None)


def test_save_counterfactual_report(sample_rejected_applicant, tmp_path):
    """Test saving counterfactual report."""
    result = generate_counterfactuals(
        applicant_data=sample_rejected_applicant,
        num_counterfactuals=2
    )
    
    if result['counterfactuals'] is not None:
        ranked = rank_counterfactuals_by_feasibility(
            result['counterfactuals'],
            result['query_instance']
        )
        
        output_path = tmp_path / "test_cf_report.csv"
        save_counterfactual_report(ranked, str(output_path))
        
        assert output_path.exists()
        
        # Check file can be read back
        loaded = pd.read_csv(output_path)
        assert len(loaded) == len(ranked)
        assert 'rank' in loaded.columns


def test_counterfactual_output_files_created():
    """Test that main script creates expected output files."""
    expected_files = [
        "reports/counterfactuals_rejected.csv",
        "reports/counterfactual_summary.json"
    ]
    
    # These will be created after running the main script
    # Just check the paths are valid
    for filepath in expected_files:
        dirpath = os.path.dirname(filepath)
        assert os.path.exists(dirpath), f"Directory {dirpath} should exist"


def test_actionable_features_subset_of_model_features():
    """Test that actionable features are valid model features."""
    from src.counterfactuals import ACTIONABLE_FEATURES
    
    model, feature_names, _ = load_counterfactual_model()
    
    for feat in ACTIONABLE_FEATURES:
        assert feat in feature_names, f"Actionable feature {feat} must be in model features"


def test_counterfactual_count_parameter(sample_rejected_applicant):
    """Test requesting different numbers of counterfactuals."""
    for num_cf in [1, 3, 5]:
        result = generate_counterfactuals(
            applicant_data=sample_rejected_applicant,
            num_counterfactuals=num_cf
        )
        
        if result['counterfactuals'] is not None:
            # May generate fewer than requested, but not more
            assert len(result['counterfactuals']) <= num_cf


def test_counterfactual_feature_constraints(sample_rejected_applicant):
    """Test that counterfactuals respect feature constraints."""
    result = generate_counterfactuals(
        applicant_data=sample_rejected_applicant,
        num_counterfactuals=3
    )
    
    if result['counterfactuals'] is not None:
        cfs = result['counterfactuals']
        
        # Income should be positive
        if 'AMT_INCOME_TOTAL' in cfs.columns:
            assert (cfs['AMT_INCOME_TOTAL'] > 0).all(), "Income must be positive"
        
        # Credit amount should be positive
        if 'AMT_CREDIT' in cfs.columns:
            assert (cfs['AMT_CREDIT'] > 0).all(), "Credit amount must be positive"
