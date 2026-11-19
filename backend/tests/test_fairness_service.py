import pytest
import pandas as pd
from backend.services.fairness_service import FairnessService

def test_fairness_service_initialization():
    """Test fairness service initializes"""
    service = FairnessService()
    assert service is not None
    assert service.fairness_threshold == 0.1

def test_create_age_bands():
    """Test age band creation from DAYS_BIRTH"""
    service = FairnessService()
    
    # Create test data: -9125 days = 25 years, -18250 days = 50 years
    days_birth = pd.Series([-9125, -12775, -16425, -20075, -25550])
    
    age_bands = service.create_age_bands(days_birth)
    
    assert len(age_bands) == 5
    assert age_bands[0] == '25-35'  # 25 years
    assert age_bands[1] == '35-45'  # 35 years
    assert age_bands[2] == '45-55'  # 45 years
    assert age_bands[3] == '55+'    # 55 years
    assert age_bands[4] == '55+'    # 70 years

def test_create_family_status_with_children():
    """Test family status creation with CNT_CHILDREN"""
    service = FairnessService()
    
    df = pd.DataFrame({
        'CNT_CHILDREN': [0, 1, 2, 0, 3]
    })
    
    family_status = service.create_family_status(df)
    
    assert family_status[0] == 'No Children'
    assert family_status[1] == 'Has Children'
    assert family_status[2] == 'Has Children'
    assert family_status[3] == 'No Children'
    assert family_status[4] == 'Has Children'

def test_create_family_status_missing_column():
    """Test family status when column is missing"""
    service = FairnessService()
    
    df = pd.DataFrame({
        'other_col': [1, 2, 3]
    })
    
    family_status = service.create_family_status(df)
    
    # Should default to No Children
    assert all(family_status == 'No Children')

def test_age_bands_edge_cases():
    """Test age band edge cases"""
    service = FairnessService()
    
    # Exact boundaries
    days_birth = pd.Series([
        -9125,  # Exactly 25 years
        -12775, # Exactly 35 years
        -16425, # Exactly 45 years
        -20075  # Exactly 55 years
    ])
    
    age_bands = service.create_age_bands(days_birth)
    
    # Check boundaries (right=False means upper bound is exclusive)
    assert age_bands[0] == '25-35'
    assert age_bands[1] == '35-45'
    assert age_bands[2] == '45-55'
    assert age_bands[3] == '55+'

def test_age_bands_young_applicants():
    """Test age bands for young applicants"""
    service = FairnessService()
    
    # Very young: 18, 22, 24 years
    days_birth = pd.Series([-6570, -8030, -8760])
    
    age_bands = service.create_age_bands(days_birth)
    
    assert all(age_bands == '<25')
