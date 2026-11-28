**Agreement of LIME With SHAP**

| Set-up | Spearman | Top-5 overlap | Sign agreement | Locality gap | Seed stability |
|---|---|---|---|---|---|
| Whole-model LIME | 0.02 | 36% | 52% | 0.00 | – |
| Local LIME (proposed) | 0.64 | 48% | 73% | 0.51 | 0.83 |

150 test applicants (50 per band), 5 seeds. Spearman on SHAP's top 25 features. Locality gap: rank correlation with the applicant's own SHAP values minus that with another applicant's. Seed stability: top-5 overlap across seeds.
