# Explainability Analysis Report

**Generated:** 2026-11-25 22:25:19  
**Model:** LightGBM  
**Test Samples:** 61,503  
**SHAP Sample Size:** 5,000  

---

## 1. Global Feature Importance

### Top 10 Features (SHAP-based)

| Rank | Feature | SHAP Importance |
|---|---|---|
| 265 | CREDIT_GOODS_EXT_RISK | 0.249328 |
| 236 | EXT_SOURCE_MEAN | 0.230456 |
| 252 | CREDIT_ANNUITY_RATIO | 0.132275 |
| 130 | NAME_EDUCATION_TYPE_Higher_education | 0.084450 |
| 9 | DAYS_ID_PUBLISH | 0.064738 |
| 105 | BUREAU_AVG_CREDIT_AMT | 0.064151 |
| 27 | EXT_SOURCE_1 | 0.062890 |
| 269 | PREV_APP_REFUSAL_RATE | 0.062526 |
| 5 | AMT_GOODS_PRICE | 0.054927 |
| 238 | EXT_SOURCE_MAX | 0.053544 |

**Key Insights:**
- Most important feature: **CREDIT_GOODS_EXT_RISK**
- SHAP importance range: 0.000000 to 0.249328
- Top 10 features account for significant prediction variance

---

## 2. SHAP Validation

### Additivity Test
- **Base value:** -2.868516
- **Max SHAP sum difference:** 0.00000000
- **Status:** ✅ Consistent (difference < 1e-6)

### Stability Test (100 samples, 2 runs)
- **Average correlation:** 1.000000
- **Minimum correlation:** 1.000000
- **Status:** ✅ Highly stable (deterministic)

---

## 3. SHAP vs LIME Consistency

Average top-10 feature overlap: **58.3%**

| Applicant | Band | Overlap | Percentage |
|---|---|---|---|
| 1 | approve | 5/10 | 50% |
| 2 | approve | 6/10 | 60% |
| 3 | review | 8/10 | 80% |
| 4 | review | 6/10 | 60% |
| 5 | reject | 5/10 | 50% |
| 6 | reject | 5/10 | 50% |

**Interpretation:**
- SHAP and LIME agree on **58%** of top features on average
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
✅ **Agreement:** 58% overlap in top features validates both methods

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
