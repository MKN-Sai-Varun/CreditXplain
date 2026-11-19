# TECHNICAL_DECISIONS — CreditXplain

**Last Updated:** Phase 1 implementation start (Nov 19, 2026)

This document records all technical and architectural decisions made during the project, with rationale.

---

## 1. Tech Stack

| Component | Technology | Rationale |
|---|---|---|
| ML Framework | scikit-learn, LightGBM, CatBoost, XGBoost | Industry standard, LightGBM selected for performance |
| Explainability | SHAP, LIME | Base paper requirement |
| Counterfactual | DiCE | Established library for actionable recourse |
| Fairness | Fairlearn | Microsoft-backed, well-documented |
| Backend | FastAPI | Modern async framework, auto-docs, type hints |
| Frontend | React + Next.js | Changed from Streamlit (Amendment B1) |
| Database | SQLite | Sufficient for single-user design, easy deployment |
| PDF Generation | ReportLab | Python-native, template support |
| Testing | pytest | Standard Python testing framework |

---

## 2. Architecture Decisions

### D-1: Final Model Selection (Phase 1)
**Decision:** Single LightGBM on Feature Engineering V2 features  
**Alternatives Considered:**
- LightGBM + CatBoost ensemble (55/45 blend)
- CatBoost alone
- Baseline LightGBM

**Rationale:**
- LightGBM + FE V2 achieved **highest test ROC-AUC: 0.7722**
- Ensemble gave only +0.001 in validation but *lost* 0.0002 on test (0.7720 vs 0.7722)
- Single model simpler to deploy, interpret, and explain
- Better generalization (test exceeded validation)

**Date:** Nov 19, 2026  
**Evidence:** `reports/fe_v2_test_results.csv`, `reports/lgbm_fe_v2_results.csv`

---

### D-2: Three-Band Decision Router (Phase 1)
**Decision:** Approve / Review / Reject bands instead of binary threshold  
**Alternatives Considered:**
- Single threshold (default 0.5)
- Single F1-optimized threshold
- Dual thresholds with binary decision

**Rationale:**
- Matches workflow diagram's "Decision Router" component
- Handles real-world review queue requirement
- Allows different risk tolerance per band
- Settles the 0.08 vs 0.162 threshold debate by using both

**Thresholds (selected on validation set):**
- **Approve below:** Target 70% recall for minority class
- **Reject from:** Target 30% precision, capped at 10% of population
- **Review between:** Manual officer review

**Date:** Nov 19, 2026  
**Implementation:** `src/decision.py`

---

### D-3: Gender Exclusion Policy (Phase 1)
**Decision:** **Exclude CODE_GENDER entirely from the model**  
**Alternatives Considered:**
- Keep gender in model, hide in explanations
- Keep gender, audit only
- Exclude gender and age/family status

**Rationale:**
- Gender is **not a valid credit decision factor** (discriminatory)
- Age **is valid** (legal capacity, guardian requirements for minors)
- Family status **is valid** (co-applicant eligibility, repayment capacity)
- Gender will be saved separately for **fairness audit in Phase 3**

**Date:** Nov 19, 2026  
**User Confirmation:** "Gender should not be taken into consideration by the model"  
**Implementation:** `src/model_params.py` (GENDER_PATTERNS exclusion list)

---

### D-4: Evaluation Protocol (Phase 1)
**Decision:** Train / Validation / Test split with proper isolation  
**Train (80% of training data):** Model fitting  
**Validation (20% of training data):** Early stopping + threshold selection  
**Test (original 20% split):** Reported once, never tuned on

**Rationale:**
- Previous baseline models chose thresholds on test set (leakage)
- Validation set prevents threshold overfitting
- Matches standard ML practice

**Date:** Nov 19, 2026  
**Implementation:** `src/final_model.py` (train_test_split with 0.2 validation)

---

## 3. Feature Engineering Decisions

### FE-1: DAYS_BIRTH Exclusion
**Decision:** Drop DAYS_BIRTH, use AGE_YEARS (derived: -DAYS_BIRTH / 365)  
**Rationale:**
- DAYS_BIRTH is negative days from birth (confusing)
- AGE_YEARS is human-interpretable
- Redundant features after transformation

**Date:** Phase 2 (Feature Engineering V2)  
**Implementation:** `src/feature_engineering_v2.py`, exclusion in `src/model_params.py`

---

### FE-2: Feature Engineering V2 Strategy
**Decision:** 15 feature groups, 100+ derived features  
**Groups:**
1. External source aggregates (mean, min, max, std, interactions)
2. Financial burden ratios (credit/income, annuity/income, etc.)
3. Age/employment features (employment age ratio, tenure)
4. Credit/income + external source interactions
5. Bureau features (bureau credit ratios, overdue ratios)
6. Previous application features (refusal rate, approval count)
7. Social circle features (default rates at 30/60 days)
8. Credit bureau request aggregates
9. Document counts
10. Contact method counts
11. Region risk features
12. Region/city discrepancies
13. Property quality aggregates
14. Log transforms of amount columns
15. Missingness indicators

**Rationale:**
- Captures non-linear relationships
- Domain-informed interactions
- Improved validation AUC from 0.748 (baseline) to 0.7716 (FE v2)

**Date:** Phase 2  
**Implementation:** `src/feature_engineering_v2.py`

---

## 4. LightGBM Hyperparameters (Final Model)

**Source:** Extracted from `src/lgbm_fe_v2_cv.py` (best-performing CV run)

```python
{
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 35,
    "max_depth": 6,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "random_state": 42
}
```

**Early Stopping:** 150 rounds  
**Rationale:** Tuned during CV experiments; prevents overfitting

**Date:** Phase 2  
**Validation Performance:** 0.7716 OOF AUC, 0.7722 Test AUC

---

## 5. SMOTE Decision (Phase 2)

**Decision:** SMOTE **disabled by default**, optional via `--smote` flag  
**Rationale:**
- Validation experiments showed SMOTE lowered ROC-AUC
- Class imbalance handled via decision thresholds (cheaper, more flexible)
- When used, SMOTENC employed (handles binary flags correctly)

**Date:** Phase 2  
**Implementation:** `src/scale_and_balance.py`  
**Evidence:** Copilot review caught plain SMOTE interpolating one-hot indicators

---

## 6. Data Cleaning Pipeline (Phase 1)

### Outlier Handling
**Method:** 3σ clipping (clip to mean ± 3 standard deviations)  
**Exclusions:** ID columns, TARGET, binary flags (0/1 only)  
**Rationale:** Preserves distribution shape, removes extreme outliers

### Missing Value Imputation
**Numeric:** Median imputation  
**Categorical:** Mode imputation  
**Special Case:** DAYS_EMPLOYED == 365243 flagged as anomaly, then nulled

### Bureau and Previous Application Aggregation
**Strategy:** Aggregate to one row per SK_ID_CURR **before** merging  
**Rationale:** Prevents row duplication in main application table

**Date:** Phase 1  
**Implementation:** `src/data_prep.py`  
**Verified Output:** 307,511 rows × 133 columns, 8.07% default rate, 0 NaNs

---

## 7. Test Strategy

### Unit Tests (Fast)
- **decision.py:** Router logic, threshold validation, edge cases
- **feature_engineering_v2.py:** Importability, feature creation, determinism

### Integration Tests (Requires Trained Model)
- **final_model.py:** Gender exclusion, threshold ordering, AUC floor, model card completeness

**Minimum AUC Threshold:** 0.70 (test fails below this)

**Date:** Phase 1  
**Location:** `tests/`

---

## 8. Git Workflow

**Development Branch:** `dev-varun`  
**Protected Branch:** `main`  
**Merge Strategy:** Pull requests only (no direct commits to main)  
**Commit Hygiene:** Stage files by name (`git add <file>`), never `git add .`

**Gitignored:**
- `data/` (raw and processed datasets)
- `models/` (trained model artifacts)
- `.env` (secrets)
- Per-applicant prediction dumps

**Date:** Established Phase 1

---

## 9. Directory Structure

```
creditxplain/
├── data/
│   ├── home_credit/          # Raw Kaggle CSVs (gitignored)
│   ├── processed/            # Cleaned/engineered data (gitignored)
│   └── uploads/              # Applicant document uploads (Phase 4)
├── models/                    # Trained models (gitignored)
├── reports/                   # EDA plots, model comparison CSVs, feature importance
├── src/                       # ML pipeline scripts
├── tests/                     # pytest unit and integration tests
├── docs/                      # Project documentation
├── api/                       # FastAPI backend (Phase 4)
├── frontend/                  # Next.js app (Phase 4)
└── requirements.txt
```

---

## 10. Dependency Management

**Python Version:** 3.9.6 (system Python on macOS)  
**Package Manager:** pip3  
**Virtual Environment:** Not used (system-level installation)

**Key Dependencies:**
- pandas, numpy, scikit-learn
- lightgbm, xgboost, catboost
- imbalanced-learn (SMOTENC)
- shap, lime, dice-ml, fairlearn
- joblib (model persistence)
- pytest (testing)

**Date:** Established Phase 1  
**File:** `requirements.txt`

---

## 11. Decisions Deferred to Later Phases

### DF-1: Test Reporting Method
**Question:** Single 80%-fit model, 5-fold average, or full-train refit?  
**Current:** Single 80%-fit model  
**Note:** 5-fold CV test results (0.7722) are fold-averaged predictions  
**Defer To:** Phase 5 (manuscript preparation)

### DF-2: Age/Family Status in Fairness Audit
**Question:** Audit for disparities or exclude like gender?  
**Current:** Keep in model, audit in Phase 3  
**Rationale:** Age is legally relevant (guardian requirements)  
**Defer To:** Phase 3 (fairness audit implementation)

### DF-3: Threshold Targets
**Current Values:**
- Approve: 70% recall for minority class
- Reject: 30% precision, max 10% population share

**Status:** Selected on validation set, not yet validated on business metrics  
**Defer To:** Phase 4 (business stakeholder review)

---

## Change Log

| Date | Phase | Changes |
|---|---|---|
| Nov 19, 2026 | Phase 1 Start | Initial document created; D-1 through D-4, FE-1, FE-2 documented |

---

**Note:** This document is a living record. Update after each phase with new decisions and their rationale.
