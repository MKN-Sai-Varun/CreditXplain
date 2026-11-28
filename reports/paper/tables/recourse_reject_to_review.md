**Recourse Evaluation, Reject to Review**

| Method | Usable | Holds on retrained | Usable and holds | Mean change | Levers | ms |
|---|---|---|---|---|---|---|
| Hybrid (deployed) | 62.0% | 38.2% | 23.7% | 0.307 | 1.91 | 95 |
| Proposed, with waiting | 62.0% | 23.1% | 14.3% | 0.217 | 1.57 | 38 |
| Proposed, loan levers only | 57.3% | 24.4% | 14.0% | 0.185 | 1.48 | 23 |
| Ensemble-robust, 5 of 5 | 40.0% | 53.3% | 21.3% | 0.210 | 1.65 | 29 |
| Ensemble-robust, 4 of 5 | 48.7% | 35.6% | 17.3% | 0.189 | 1.53 | 31 |
| Ensemble-robust, with waiting | 46.7% | 47.1% | 22.0% | 0.284 | 1.86 | 58 |
| Proposed, margin 0.2 | 48.3% | 34.5% | 16.7% | 0.206 | 1.68 | 22 |
| Ablation: no margin | 10.0% | 13.3% | 4.3% | 0.148 | 1.40 | 24 |
| Ablation: no plausibility | 53.3% | 20.5% | 13.0% | 0.249 | 1.42 | 26 |
| Random search | 59.7% | 29.6% | 17.7% | 0.269 | 3.00 | 19 |
| Baseline engine | 22.7% | 20.1% | 8.7% | 0.255 | 1.38 | 15 |
| DiCE (20 applicants) | 20.0% | 50.0% | 15.0% | 0.386 | 1.80 | 216 |

Usable: found, plausible and clears the safety margin (share of applicants). Holds on retrained: share of found options that reach the promised band on all 5 held-out bootstrap models. 300 applicants per method, DiCE 20.
