"""
Three-band decision router: Approve / Review / Reject
Validates inputs and assigns applicants to bands based on risk probability.
"""
import numpy as np


def decide(probabilities, approve_below=0.08, reject_from=0.20):
    """
    Classify risk probabilities into three decision bands.
    
    Args:
        probabilities: array of default probabilities (0-1)
        approve_below: threshold for automatic approval (low risk)
        reject_from: threshold for automatic rejection (high risk)
    
    Returns:
        array of decisions: "approve", "review", or "reject"
    
    Raises:
        ValueError: if thresholds are invalid
    """
    if not 0 <= approve_below < reject_from <= 1:
        raise ValueError(
            f"Thresholds must satisfy 0 <= approve_below < reject_from <= 1. "
            f"Got approve_below={approve_below}, reject_from={reject_from}"
        )
    
    probs = np.asarray(probabilities)
    
    if not np.all((probs >= 0) & (probs <= 1)):
        raise ValueError("Probabilities must be in range [0, 1]")
    
    decisions = np.full(len(probs), "review", dtype=object)
    decisions[probs < approve_below] = "approve"
    decisions[probs >= reject_from] = "reject"
    
    return decisions


def decide_one(probability, approve_below=0.08, reject_from=0.20):
    """
    Classify a single applicant's risk probability.
    
    Args:
        probability: default probability (0-1)
        approve_below: threshold for automatic approval
        reject_from: threshold for automatic rejection
    
    Returns:
        "approve", "review", or "reject"
    """
    result = decide([probability], approve_below, reject_from)
    return result[0]
