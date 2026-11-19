import pytest
from backend.services.loan_calculator import LoanCalculator

def test_loan_calculator_initialization():
    """Test loan calculator initializes"""
    calculator = LoanCalculator()
    assert calculator is not None
    assert calculator.base_rate == 0.10  # 10% base rate

def test_calculate_interest_rate_approve():
    """Test interest rate for APPROVE decision"""
    calculator = LoanCalculator()
    
    # Low risk APPROVE (prob = 0.01)
    rate = calculator.calculate_interest_rate("APPROVE", 0.01)
    assert 0.10 <= rate <= 0.15  # Base (10%) + spread (2%) + adjustment
    
    # Higher risk APPROVE (prob = 0.03)
    rate = calculator.calculate_interest_rate("APPROVE", 0.03)
    assert 0.10 <= rate <= 0.15

def test_calculate_interest_rate_review():
    """Test interest rate for REVIEW decision"""
    calculator = LoanCalculator()
    
    # Low REVIEW (prob = 0.05)
    rate = calculator.calculate_interest_rate("REVIEW", 0.05)
    assert 0.12 <= rate <= 0.20  # Base (10%) + spread (5%) + adjustment
    
    # High REVIEW (prob = 0.10)
    rate = calculator.calculate_interest_rate("REVIEW", 0.10)
    assert 0.12 <= rate <= 0.20

def test_calculate_interest_rate_reject():
    """Test interest rate for REJECT decision"""
    calculator = LoanCalculator()
    
    rate = calculator.calculate_interest_rate("REJECT", 0.15)
    assert 0.15 <= rate <= 0.25  # Base (10%) + spread (8%) + adjustment

def test_calculate_emi_standard():
    """Test EMI calculation with standard values"""
    calculator = LoanCalculator()
    
    # 100,000 at 12% for 12 months
    emi, total = calculator.calculate_emi(100000, 0.12, 12)
    
    # Expected: ~8,884.88 per month
    assert 8800 <= emi <= 8900
    assert 106000 <= total <= 107000  # Total ~106,619

def test_calculate_emi_zero_interest():
    """Test EMI with zero interest rate"""
    calculator = LoanCalculator()
    
    emi, total = calculator.calculate_emi(120000, 0.0, 12)
    
    assert emi == 10000.0  # 120000 / 12
    assert total == 120000.0

def test_calculate_emi_different_tenures():
    """Test EMI with different loan tenures"""
    calculator = LoanCalculator()
    
    # 6 months
    emi_6, total_6 = calculator.calculate_emi(100000, 0.12, 6)
    
    # 24 months
    emi_24, total_24 = calculator.calculate_emi(100000, 0.12, 24)
    
    # Shorter tenure = higher EMI, lower total interest
    assert emi_6 > emi_24
    assert total_6 < total_24

def test_calculate_emi_invalid_inputs():
    """Test EMI calculation with invalid inputs"""
    calculator = LoanCalculator()
    
    with pytest.raises(ValueError):
        calculator.calculate_emi(100000, 0.12, 0)  # Zero tenure
    
    with pytest.raises(ValueError):
        calculator.calculate_emi(0, 0.12, 12)  # Zero principal

def test_calculate_loan_terms_approve():
    """Test complete loan terms for APPROVE"""
    calculator = LoanCalculator()
    
    terms = calculator.calculate_loan_terms(
        loan_amount=100000,
        decision="APPROVE",
        probability=0.02,
        tenure_months=12
    )
    
    assert terms['loan_amount'] == 100000
    assert 0.10 <= terms['interest_rate'] <= 0.15
    assert terms['tenure_months'] == 12
    assert terms['monthly_emi'] > 0
    assert terms['total_payable'] > terms['loan_amount']
    assert terms['total_interest'] > 0

def test_calculate_loan_terms_review():
    """Test complete loan terms for REVIEW"""
    calculator = LoanCalculator()
    
    terms = calculator.calculate_loan_terms(
        loan_amount=50000,
        decision="REVIEW",
        probability=0.08,
        tenure_months=24
    )
    
    assert terms['loan_amount'] == 50000
    assert 0.12 <= terms['interest_rate'] <= 0.20
    assert terms['tenure_months'] == 24
    assert terms['monthly_emi'] > 0

def test_interest_rate_increases_with_risk():
    """Test that interest rate increases with default probability"""
    calculator = LoanCalculator()
    
    # APPROVE with low vs high probability
    rate_low = calculator.calculate_interest_rate("APPROVE", 0.01)
    rate_high = calculator.calculate_interest_rate("APPROVE", 0.03)
    
    assert rate_high >= rate_low  # Higher risk = higher rate
    
    # REVIEW with low vs high probability
    rate_low_review = calculator.calculate_interest_rate("REVIEW", 0.04)
    rate_high_review = calculator.calculate_interest_rate("REVIEW", 0.11)
    
    assert rate_high_review >= rate_low_review

def test_interest_rate_capped_at_maximum():
    """Test that interest rate is capped at 25%"""
    calculator = LoanCalculator()
    
    # Very high risk should be capped
    rate = calculator.calculate_interest_rate("REJECT", 0.50)
    
    assert rate <= 0.25  # Maximum 25%
