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


def select_cutoffs(val_probs, y_val, max_approve_default=0.03, min_reject_default=0.20):
    """
    Picks the two cutoffs on a validation set (the rule used for the final model):
    - approve below the lowest of the 35th, 37th, ..., 49th percentiles whose
      band has a default rate of at most 3% (40th percentile if none does);
    - reject from the 80th percentile, or the 85th if that band's default rate
      is under 20%.
    Returns (approve_below, reject_from).
    """
    probs = np.asarray(val_probs, dtype=float)
    y = np.asarray(y_val, dtype=float)

    approve_below = None
    for percentile in range(35, 50, 2):
        candidate = np.percentile(probs, percentile)
        mask = probs < candidate
        if mask.sum() > 0 and y[mask].mean() <= max_approve_default:
            approve_below = candidate
            break
    if approve_below is None:
        approve_below = np.percentile(probs, 40)

    reject_from = np.percentile(probs, 80)
    mask = probs >= reject_from
    if mask.sum() > 0 and y[mask].mean() < min_reject_default:
        reject_from = np.percentile(probs, 85)

    return float(approve_below), float(reject_from)
