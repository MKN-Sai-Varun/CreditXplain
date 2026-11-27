**Removing Age From the Model**

| Variant | ROC-AUC | Approve | Default if approved | Age DPD | Equal opp. gap |
|---|---|---|---|---|---|
| Current (age kept) | 0.7668 | 34.8% | 2.09% | 0.349 | 0.349 |
| Without direct age | 0.7652 | 34.7% | 2.19% | 0.364 | 0.365 |
| Age-free | 0.7646 | 34.9% | 2.20% | 0.377 | 0.380 |
| Per-band cutoffs (uses age) | 0.7668 | 34.6% | 2.12% | 0.022 | 0.041 |

Run before cleaning statistics were restricted to training rows (Phase 5.6b); the current model's audit gives DPD 0.356.
