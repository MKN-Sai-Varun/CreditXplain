"""
Unit tests for the three-band decision router.
"""
import pytest
import numpy as np
from decision import decide, decide_one


def test_decide_validates_threshold_order():
    with pytest.raises(ValueError, match="must satisfy"):
        decide([0.5], approve_below=0.3, reject_from=0.2)


def test_decide_validates_threshold_range():
    with pytest.raises(ValueError, match="must satisfy"):
        decide([0.5], approve_below=-0.1, reject_from=0.5)
    
    with pytest.raises(ValueError, match="must satisfy"):
        decide([0.5], approve_below=0.5, reject_from=1.2)


def test_decide_validates_probabilities():
    with pytest.raises(ValueError, match="must be in range"):
        decide([0.5, 1.5], approve_below=0.1, reject_from=0.3)
    
    with pytest.raises(ValueError, match="must be in range"):
        decide([-0.1, 0.5], approve_below=0.1, reject_from=0.3)


def test_decide_assigns_correct_bands():
    probs = [0.05, 0.10, 0.15, 0.20, 0.25]
    decisions = decide(probs, approve_below=0.10, reject_from=0.20)
    
    # 0.05 < 0.10 -> approve
    # 0.10 >= 0.10 and < 0.20 -> review
    # 0.15 >= 0.10 and < 0.20 -> review
    # 0.20 >= 0.20 -> reject
    # 0.25 >= 0.20 -> reject
    assert list(decisions) == ["approve", "review", "review", "reject", "reject"]


def test_decide_handles_edge_cases():
    # Boundary behavior: value exactly at threshold goes to the higher band
    # 0.10 >= approve_below (0.10) -> review (not approve)
    # 0.20 >= reject_from (0.20) -> reject (not review)
    decisions = decide([0.10, 0.20], approve_below=0.10, reject_from=0.20)
    assert list(decisions) == ["review", "reject"]


def test_decide_handles_array_input():
    probs = np.array([0.05, 0.15, 0.25])
    decisions = decide(probs, approve_below=0.10, reject_from=0.20)
    assert len(decisions) == 3


def test_decide_one():
    assert decide_one(0.05, approve_below=0.10, reject_from=0.20) == "approve"
    assert decide_one(0.15, approve_below=0.10, reject_from=0.20) == "review"
    assert decide_one(0.25, approve_below=0.10, reject_from=0.20) == "reject"
