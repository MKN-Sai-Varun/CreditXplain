# Counterfactual Recommendations Report

**Generated:** 2026-11-19 08:47:40  
**Analyzed:** 10 rejected applicants  
**Model:** LightGBM  

---

## Executive Summary

- **Total counterfactuals generated:** 30
- **Successful (flip to APPROVE):** 1 (3.3%)
- **Average changes required:** 1.5 features
- **Feasibility:** 0.1 counterfactuals per rejected applicant

---

## Key Findings

### Most Frequently Changed Features

Features that most often need adjustment to secure approval:

1. **INCOME_PER_FAMILY_MEMBER** (4 times, 13.3% of counterfactuals)
2. **OWN_CAR_AGE** (4 times, 13.3% of counterfactuals)
3. **CONTACT_METHOD_COUNT** (3 times, 10.0% of counterfactuals)
4. **AMT_CREDIT** (3 times, 10.0% of counterfactuals)
5. **CREDIT_PER_FAMILY_MEMBER** (3 times, 10.0% of counterfactuals)
6. **DOCUMENT_COUNT** (2 times, 6.7% of counterfactuals)
7. **FLAG_WORK_PHONE** (2 times, 6.7% of counterfactuals)
8. **GOODS_INCOME_RATIO** (2 times, 6.7% of counterfactuals)
9. **CREDIT_TO_GOODS_RATIO** (2 times, 6.7% of counterfactuals)
10. **GOODS_CREDIT_RATIO** (2 times, 6.7% of counterfactuals)


---

## Actionability Assessment

**Highly Actionable (Easy to Change):**
- Loan amount (AMT_CREDIT)
- Co-applicant addition (CNT_FAM_MEMBERS, income)
- Document updates (DAYS_ID_PUBLISH)
- Contact information (phone, email flags)

**Moderately Actionable (Time Required):**
- Income increase (AMT_INCOME_TOTAL) - new job, promotion, side income
- Employment duration (DAYS_EMPLOYED) - wait/maintain employment
- Car ownership (FLAG_OWN_CAR_Y) - purchase asset

**Least Actionable (Difficult/Slow):**
- External credit scores (EXT_SOURCE_*) - improve credit history over months/years
- Credit ratios derived from external scores

---

## Recommendations for Loan Officers

When providing recourse to rejected applicants:

1. **Prioritize feasible changes:** Start with counterfactuals requiring fewest modifications
2. **Focus on financial levers:** Loan amount and income are most actionable
3. **Set realistic timelines:** Some changes (employment duration) require waiting
4. **Combine strategies:** Multiple small changes may be easier than one large change

---

## Sample Counterfactuals


### Example 1: Applicant 21024

**Current Status:**
- Risk: 13.02%
- Decision: REJECT

**Recommended Changes:**
- AMT_GOODS_PRICE: 238500.00 → 925278.10 (+288.0%)
- CONTACT_METHOD_COUNT: 2.00 → 2.10 (+5.0%)

**Outcome:**
- New risk: 11.74%
- New decision: APPROVE ✅

---

## Limitations

1. **Static Analysis:** Counterfactuals assume other factors remain constant
2. **External Scores:** Changes to external credit scores are suggested but require time and actual credit behavior changes
3. **Correlation vs Causation:** Model correlations may not reflect true causal relationships
4. **Individual Variation:** What works for one applicant may not work for another

---

## Next Steps (Phase 4)

- Integrate counterfactuals into applicant portal
- Provide interactive "what-if" calculator
- Show top 3 most feasible paths to approval
- Include timeline estimates for each recommendation

---

**Report Complete**
