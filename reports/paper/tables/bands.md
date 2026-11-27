**Decision Bands on the Test Set**

| Band | Rule | Share | Actual default | Mean predicted |
|---|---|---|---|---|
| Approve | p < 0.0351 | 34.9% | 2.1% | 2.1% |
| Review | between | 44.9% | 6.7% | 6.6% |
| Reject | p ≥ 0.1190 | 20.2% | 21.4% | 21.2% |

Cutoffs chosen on the validation set (src/decision.py).
