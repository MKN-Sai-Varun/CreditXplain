"""
Tests for feature_engineering_v2.py
"""
import pytest
import numpy as np
import pandas as pd
from feature_engineering_v2 import add_features


def test_add_features_is_importable():
    """Verify the module can be imported without executing pipeline code."""
    assert callable(add_features)


def test_add_features_creates_expected_features():
    """Check that key feature groups are created."""
    df = pd.DataFrame({
        "EXT_SOURCE_1": [0.5, 0.6],
        "EXT_SOURCE_2": [0.4, 0.7],
        "EXT_SOURCE_3": [0.3, 0.8],
        "AMT_CREDIT": [100000, 200000],
        "AMT_INCOME_TOTAL": [50000, 80000],
        "AMT_ANNUITY": [5000, 8000],
        "AGE_YEARS": [30, 45],
    })
    
    result = add_features(df)
    
    assert "EXT_SOURCE_MEAN" in result.columns
    assert "CREDIT_INCOME_RATIO_NEW" in result.columns
    assert "ANNUITY_INCOME_RATIO" in result.columns
    assert "AGE_SQUARED" in result.columns


def test_add_features_handles_missing_columns():
    """Verify it doesn't crash when optional columns are missing."""
    df = pd.DataFrame({
        "AMT_CREDIT": [100000],
        "AMT_INCOME_TOTAL": [50000],
    })
    
    result = add_features(df)
    assert len(result) == 1


def test_add_features_replaces_infinities():
    """Ensure division by zero produces NaN, not inf."""
    df = pd.DataFrame({
        "AMT_CREDIT": [100000],
        "AMT_INCOME_TOTAL": [0],  # Will cause division by zero
    })
    
    result = add_features(df)
    assert not np.isinf(result.values).any()


def test_add_features_is_deterministic():
    """Same input should produce same output."""
    df = pd.DataFrame({
        "EXT_SOURCE_1": [0.5],
        "EXT_SOURCE_2": [0.4],
        "AMT_CREDIT": [100000],
        "AMT_INCOME_TOTAL": [50000],
    })
    
    result1 = add_features(df.copy())
    result2 = add_features(df.copy())
    
    pd.testing.assert_frame_equal(result1, result2)
