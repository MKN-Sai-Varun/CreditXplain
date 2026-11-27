**Agreement of LIME With SHAP**

| Set-up | Spearman | Top-5 overlap | Sign agreement | Locality gap | Seed stability |
|---|---|---|---|---|---|
| Whole-model LIME (previous) | 0.02 | 36% | 52% | 0.00 | – |
| Local LIME (proposed) | 0.64 | 48% | 73% | 0.51 | 0.83 |

150 test applicants (50 per band), 5 seeds. Locality gap: how much better the local surrogate fits the applicant's neighbourhood than the whole data.
