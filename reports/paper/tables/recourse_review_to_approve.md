**Recourse Evaluation, Review to Approve**

| Method | Usable | Holds on retrained | Usable and holds | Mean change | Levers | ms |
|---|---|---|---|---|---|---|
| Hybrid (deployed) | 45.3% | 24.3% | 11.0% | 0.379 | 2.07 | 86 |
| Proposed, with waiting | 45.3% | 11.0% | 5.0% | 0.286 | 1.76 | 34 |
| Proposed, loan levers only | 37.0% | 11.7% | 4.3% | 0.207 | 1.53 | 20 |
| Ensemble-robust, 5 of 5 | 17.7% | 43.4% | 7.7% | 0.249 | 1.72 | 24 |
| Ensemble-robust, 4 of 5 | 27.0% | 24.7% | 6.7% | 0.226 | 1.62 | 26 |
| Ensemble-robust, with waiting | 25.7% | 39.0% | 10.0% | 0.401 | 2.14 | 52 |
| Proposed, margin 0.2 | 24.7% | 33.8% | 8.3% | 0.234 | 1.77 | 18 |
| Ablation: no margin | 8.0% | 7.9% | 0.7% | 0.182 | 1.56 | 22 |
| Ablation: no plausibility | 36.0% | 7.9% | 4.0% | 0.287 | 1.56 | 22 |
| Random search | 37.7% | 23.0% | 8.7% | 0.309 | 3.00 | 19 |
| Baseline engine | 12.0% | 7.9% | 1.7% | 0.277 | 1.54 | 15 |
| DiCE (20 applicants) | 5.0% | 20.0% | 0.0% | 0.512 | 2.00 | 126 |

Usable: found, plausible and clears the safety margin (share of applicants). Holds on retrained: share of found options that reach the promised band on all 5 held-out bootstrap models. 300 applicants per method, DiCE 20.
