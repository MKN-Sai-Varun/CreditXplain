# Explainability Analysis Report

**Generated:** 2026-11-27 14:46:14  
**Model:** LightGBM  
**Test Samples:** 61,503  
**SHAP Sample Size:** 5,000  

---

## 1. Global Feature Importance

### Top 10 Features (SHAP-based)

| Rank | Feature | SHAP Importance |
|---|---|---|
| 265 | CREDIT_GOODS_EXT_RISK | 0.254380 |
| 236 | EXT_SOURCE_MEAN | 0.229018 |
| 252 | CREDIT_ANNUITY_RATIO | 0.130137 |
| 130 | NAME_EDUCATION_TYPE_Higher_education | 0.081187 |
| 269 | PREV_APP_REFUSAL_RATE | 0.063430 |
| 9 | DAYS_ID_PUBLISH | 0.062437 |
| 27 | EXT_SOURCE_1 | 0.060063 |
| 105 | BUREAU_AVG_CREDIT_AMT | 0.059521 |
| 5 | AMT_GOODS_PRICE | 0.053390 |
| 248 | EXT_SOURCE_2_3_PRODUCT | 0.051755 |

**Key Insights:**
- Most important feature: **CREDIT_GOODS_EXT_RISK**
- SHAP importance range: 0.000000 to 0.254380
- Top 10 features account for significant prediction variance

---

## 2. SHAP Validation

### Additivity Test
- **Base value:** -2.864420
- **Max SHAP sum difference:** 0.00000000
- **Status:** ✅ Consistent (difference < 1e-6)

### Stability Test (100 samples, 2 runs)
- **Average correlation:** 1.000000
- **Minimum correlation:** 1.000000
- **Status:** ✅ Highly stable (deterministic)

---

## 3. SHAP vs LIME Consistency

Average top-10 feature overlap: **63.3%**

| Applicant | Band | Overlap | Percentage |
|---|---|---|---|
| 1 | approve | 5/10 | 50% |
| 2 | approve | 8/10 | 80% |
| 3 | review | 5/10 | 50% |
| 4 | review | 6/10 | 60% |
| 5 | reject | 7/10 | 70% |
| 6 | reject | 7/10 | 70% |

**Interpretation:**
- SHAP and LIME agree on **63%** of top features on average
- ⚠️ Moderate consistency between methods
- Both methods identify similar important features

---

## 4. Local Explanations Generated

**SHAP Waterfall Plots:** 6 applicants  
**LIME Explanations:** 6 applicants  

### Sample Breakdown
- **Approve:** 2 applicants
- **Review:** 2 applicants
- **Reject:** 2 applicants

---

## 5. Files Generated

### SHAP Outputs
- `reports/shap/global_feature_importance.csv` - Feature importance table
- `reports/shap/summary_beeswarm.png` - Global SHAP summary plot
- `reports/shap/feature_importance_bar.png` - Feature importance bar chart
- `reports/shap/waterfall_applicant_*.png` - 6 individual waterfall plots

### LIME Outputs
- `reports/lime/explanation_applicant_*.html` - 6 interactive HTML explanations
- `reports/lime/explanation_applicant_*.png` - 6 static plots

### Comparison
- `reports/shap_lime_consistency.csv` - SHAP vs LIME consistency metrics
- `reports/explainability_report.md` - This report

---

## 6. Conclusions

### Explainability Quality
✅ **SHAP:** Consistent, stable, and mathematically grounded  
✅ **LIME:** Provides complementary local explanations  
✅ **Agreement:** 63% overlap in top features validates both methods

### Recommendation
Primary explainability method: **SHAP** (TreeExplainer)  
- Exact for tree-based models
- Consistent and deterministic
- Provides both global and local explanations

Secondary validation: **LIME**  
- Confirms key features independently
- Useful for non-technical stakeholders (simpler HTML output)

---

## 7. Next Steps (Phase 3)

- Use SHAP explanations in counterfactual engine (identify actionable features)
- Integrate SHAP into frontend dashboard (show explanations to officers)
- Use feature importance for fairness audit (check if sensitive features have high importance)

---

**Report Complete**
