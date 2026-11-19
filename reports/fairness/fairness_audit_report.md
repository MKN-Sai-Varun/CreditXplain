# Fairness and Bias Audit Report

**Generated:** 2026-11-19 09:05:00  
**Test Samples:** 61,503  
**Overall Approve Rate:** 34.84%  
**Overall Reject Rate:** 20.11%  

---

## Executive Summary

This report analyzes demographic disparities in model predictions and lending decisions.

**Key Findings:**

- **Largest approval disparity:** age_band (34.9% difference between groups)
- **Average demographic parity difference:** 0.1968


---

## Methodology

### Metrics Used

**Demographic Parity Difference (DPD)**
- Measures difference in positive prediction rates between groups
- Range: [-1, 1], with 0 indicating perfect parity
- Interpretation: |DPD| < 0.1 is generally considered acceptable

**Equalized Odds Difference (EOD)**
- Measures difference in true positive and false positive rates
- Ensures similar error rates across groups
- Range: [-1, 1], with 0 indicating perfect equality

**Group-Level Metrics**
- Approval, Review, and Reject rates by demographic group
- Actual default rates by group (ground truth)
- Model performance (AUC) by group

---

## Results by Demographic


### Age Band

| Group | Count | Approve | Review | Reject | Actual Default |
|---|---|---|---|---|---|
| 45-55 | 13,933 | 38.8% | 45.7% | 15.6% | 6.8% |
| 35-45 | 16,632 | 34.9% | 44.9% | 20.2% | 8.1% |
| 25-35 | 15,244 | 24.7% | 44.9% | 30.4% | 10.8% |
| 55+ | 12,484 | 48.2% | 44.3% | 7.4% | 5.2% |
| <25 | 3,210 | 13.4% | 46.9% | 39.8% | 11.4% |

**Disparity:** 34.9% difference in approval rates
🔴 **Assessment:** High disparity (> 10%)


### Has Children

| Group | Count | Approve | Review | Reject | Actual Default |
|---|---|---|---|---|---|
| With Children | 18,386 | 31.7% | 44.8% | 23.5% | 9.0% |
| No Children | 43,117 | 36.2% | 45.2% | 18.7% | 7.7% |

**Disparity:** 4.5% difference in approval rates
✅ **Assessment:** Low disparity (< 5%)


---

## Gender Monitoring

**Note:** Gender features were **excluded from the model** per ethical policy (Phase 1, Decision D-3).

Gender is monitored for outcome disparities but **does not influence predictions**.



---

## Interpretations and Recommendations

### Age

Age is **included in the model** as a legally relevant factor (guardian requirements for minors).

**Observed Patterns:**
- Younger applicants (<25) may face higher rejection due to limited credit history
- Middle-aged applicants (35-45) typically have highest approval rates (established income/credit)
- Older applicants (55+) may see lower approval if nearing retirement

**Recommendation:** Age disparities are expected and legally justified. Monitor for extreme disparities.

### Family Status

Family status features (CNT_CHILDREN, CNT_FAM_MEMBERS) are **included** as they affect repayment capacity.

**Observed Patterns:**
- Larger families may need higher income to maintain same credit ratios
- Single applicants may have more disposable income but less stability

**Recommendation:** Monitor for disproportionate impact on families with children.

### Gender (Monitoring Only)

Gender is **excluded from the model** but monitored for outcome disparities.

**If disparities exist:**
- Likely due to correlated factors (income gaps, employment differences)
- Consider additional fairness interventions (e.g., income adjustment factors)
- Document and report openly

---

## Fairness Assessment

🔴 **REQUIRES INTERVENTION**

The model shows significant disparities (> 10% approval difference). Fairness interventions needed.


---

## Limitations

1. **Small Sample Sizes:** Some demographic subgroups have limited samples, reducing statistical confidence
2. **Intersectionality:** This analysis examines one demographic at a time, not intersections (e.g., young women)
3. **Unmeasured Factors:** Disparities may reflect unmeasured legitimate factors (wealth, assets, debt-to-income)
4. **Static Analysis:** Real-world lending involves dynamic factors not captured in historical data

---

## Next Steps (Phase 4)

- Integrate fairness monitoring into live decision dashboard
- Alert officers when decisions create demographic disparities
- Provide officers with fairness context for manual reviews
- Document fairness considerations in PDF reports
- Periodic re-audit (quarterly) as new data accumulates

---

**Report Complete**
