"""
Loan Calculator Service - Calculate interest rates and EMI
"""
import math
from typing import Dict, Tuple
from backend.config import settings

class LoanCalculator:
    """Service for calculating loan terms based on risk assessment"""
    
    def __init__(self):
        self.base_rate = settings.BASE_INTEREST_RATE
        self.risk_spreads = settings.RISK_RATE_SPREAD
    
    def calculate_interest_rate(self, decision: str, probability: float) -> float:
        """
        Calculate interest rate based on decision and default probability
        
        Args:
            decision: APPROVE, REVIEW, or REJECT
            probability: Default probability (0-1)
            
        Returns:
            Annual interest rate (e.g., 0.12 for 12%)
        """
        # Base rate + risk spread based on decision band
        risk_spread = self.risk_spreads.get(decision, 0.05)
        interest_rate = self.base_rate + risk_spread
        
        # Additional adjustment within band based on exact probability
        if decision == "APPROVE":
            # APPROVE: prob < APPROVE_THRESHOLD
            # Scale from 0% to +2% within band
            prob_normalized = probability / settings.APPROVE_THRESHOLD
            prob_adjustment = prob_normalized * 0.02  # 0-2% additional
        elif decision == "REVIEW":
            # REVIEW: APPROVE_THRESHOLD <= prob < REJECT_THRESHOLD
            # Scale from 0% to +3% within band
            band_size = settings.REJECT_THRESHOLD - settings.APPROVE_THRESHOLD
            prob_in_band = probability - settings.APPROVE_THRESHOLD
            prob_normalized = prob_in_band / band_size
            prob_adjustment = prob_normalized * 0.03  # 0-3% additional
        else:  # REJECT
            # REJECT: prob >= REJECT_THRESHOLD
            # Higher rate for manual override cases
            prob_adjustment = 0.03  # Fixed +3% additional
        
        final_rate = interest_rate + prob_adjustment
        
        # Cap at reasonable maximum (25%)
        final_rate = min(final_rate, 0.25)
        
        return round(final_rate, 4)
    
    def calculate_emi(
        self, 
        principal: float, 
        annual_rate: float, 
        tenure_months: int
    ) -> Tuple[float, float]:
        """
        Calculate Equated Monthly Installment (EMI)
        
        Formula: EMI = P × r × (1 + r)^n / ((1 + r)^n - 1)
        Where:
            P = Principal loan amount
            r = Monthly interest rate (annual_rate / 12)
            n = Loan tenure in months
        
        Args:
            principal: Loan amount
            annual_rate: Annual interest rate (e.g., 0.12 for 12%)
            tenure_months: Loan tenure in months
            
        Returns:
            Tuple of (monthly_emi, total_payable)
        """
        if tenure_months <= 0:
            raise ValueError("Tenure must be positive")
        
        if principal <= 0:
            raise ValueError("Principal must be positive")
        
        # Handle zero interest rate (special case)
        if annual_rate == 0:
            monthly_emi = principal / tenure_months
            total_payable = principal
            return round(monthly_emi, 2), round(total_payable, 2)
        
        # Calculate monthly interest rate
        monthly_rate = annual_rate / 12
        
        # EMI formula
        numerator = principal * monthly_rate * math.pow(1 + monthly_rate, tenure_months)
        denominator = math.pow(1 + monthly_rate, tenure_months) - 1
        
        monthly_emi = numerator / denominator
        total_payable = monthly_emi * tenure_months
        
        return round(monthly_emi, 2), round(total_payable, 2)
    
    def calculate_loan_terms(
        self,
        loan_amount: float,
        decision: str,
        probability: float,
        tenure_months: int = 12
    ) -> Dict[str, float]:
        """
        Calculate complete loan terms
        
        Args:
            loan_amount: Requested loan amount
            decision: APPROVE, REVIEW, or REJECT
            probability: Default probability
            tenure_months: Loan tenure (default 12 months)
            
        Returns:
            Dictionary with loan terms:
            - loan_amount: Principal amount
            - interest_rate: Annual interest rate
            - tenure_months: Loan tenure
            - monthly_emi: Monthly installment
            - total_payable: Total amount to be paid
            - total_interest: Total interest amount
        """
        # Calculate interest rate
        interest_rate = self.calculate_interest_rate(decision, probability)
        
        # Calculate EMI
        monthly_emi, total_payable = self.calculate_emi(
            principal=loan_amount,
            annual_rate=interest_rate,
            tenure_months=tenure_months
        )
        
        # Calculate total interest
        total_interest = total_payable - loan_amount
        
        return {
            "loan_amount": round(loan_amount, 2),
            "interest_rate": interest_rate,
            "tenure_months": tenure_months,
            "monthly_emi": monthly_emi,
            "total_payable": total_payable,
            "total_interest": round(total_interest, 2)
        }

# Singleton instance
_loan_calculator = None

def get_loan_calculator() -> LoanCalculator:
    """Get or create loan calculator singleton"""
    global _loan_calculator
    if _loan_calculator is None:
        _loan_calculator = LoanCalculator()
    return _loan_calculator


def loan_terms_for(application, decision):
    """
    LoanTerms row for an approved application, priced on the model's risk band
    (so an officer override of a Review or Reject case pays that band's rate).
    """
    from backend.models import LoanTerms

    amount = application.features_json.get("AMT_CREDIT")
    if not amount or amount <= 0:
        raise ValueError("Application has no valid requested loan amount (AMT_CREDIT)")
    terms = get_loan_calculator().calculate_loan_terms(
        loan_amount=float(amount),
        decision=decision.decision.value,
        probability=decision.probability,
        tenure_months=application.tenure_months,
    )
    return LoanTerms(
        decision_id=decision.id,
        loan_amount=terms["loan_amount"],
        interest_rate=terms["interest_rate"],
        tenure_months=terms["tenure_months"],
        monthly_emi=terms["monthly_emi"],
        total_payable=terms["total_payable"],
    )
