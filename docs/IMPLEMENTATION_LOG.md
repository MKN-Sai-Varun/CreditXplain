# IMPLEMENTATION_LOG — CreditXplain

**Chronological Record of Implementation Work**

This document tracks what was built, when, and what results were achieved. Each entry includes commit references where applicable.

---

## Phase 1: Literature Review & Dataset Prep

### 2024-09-25: Data Preprocessing Pipeline

**Implemented:** `src/data_prep.py`  
**Author:** Member 1 (Varun) and Pranav  
**Commits:** `d264156` (dev-pranav), merged via PR #2

**What It Does:**
- Loads Home Credit CSVs: application_train, bureau, previous_application
- Aggregates bureau and previous_application to one row per SK_ID_CURR
- Cleans: flags DAYS_EMPLOYED == 365243 anomaly, median/mode imputation, 3σ clipping
- Engineers: CREDIT_TO_GOODS_RATIO, AGE_YEARS, CREDIT_INCOME_RATIO
- Simulates application timestamps (for dashboard analytics)

**Verified Output (real data run):**
- Shape: 307,511 rows × 133 columns
- Target rate: 8.07%
- Missing values: 0
- No leaked 365243 values

**Status:** ✅ Complete and verified

---

### Status: EDA Report

**Expected:** `reports/eda_age_distribution.png`, descriptive statistics  
**Status:** ⬜ Not completed (no EDA output files found in repo)  
**Gap:** Phase 1 deliverable missing

---

## Phase 2: Baseline Models & Explainability

### 2024-09-25: Baseline Model Comparison

**Implemented:** `src/train_baseline.py`  
**Author:** Varun  
**Commit:** `799fc07` (initial), refined in later commits

**What It Does:**
- 80/20 stratified split
- SMOTE on training data
- Trains Logistic Regression, Random Forest, XGBoost, LightGBM
- F1-optimal threshold selection (per model)
- Saves models to `models/`, results to `reports/baseline_model_comparison.csv`

**Results (real data):**

| Model | ROC-AUC | Precision | Recall | F1 | Best Threshold |
|---|---|---|---|---|---|
| Logistic Regression | 0.7219 | 0.2167 | 0.4068 | 0.2828 | 0.1586 |
| Random Forest | 0.7031 | 0.1854 | 0.4006 | 0.2535 | 0.2150 |
| XGBoost | 0.7467 | 0.2246 | 0.4354 | 0.2964 | 0.1515 |
| **LightGBM (selected)** | **0.7477** | **0.2367** | **0.4042** | **0.2986** | **0.1622** |

**Decision:** LightGBM selected as primary model  
**Status:** ✅ Complete and verified

---

### 2024-09-25: Scaling and Balancing Pipeline

**Implemented:** `src/scale_and_balance.py`  
**Author:** Pranav, refined by Varun  
**Commits:** `1bffbe2` (initial), `69b13b9` (fix + tests), merged via PR #3

**What It Does:**
- Drops DAYS_BIRTH (redundant with AGE_YEARS)
- Scales continuous features (StandardScaler)
- Optional SMOTENC (--smote flag, off by default)
- Saves scaler and continuous column list to `models/`

**Tests:** `tests/test_scale_and_balance.py` (3 passed on real data)

**Verified Output:**
- Train: (246,008, 237), positive rate 0.0807
- Test: (61,503, 237)

**Copilot Issues Addressed:**
- ✅ Fixed clean-checkout crash (missing `os.makedirs`)
- ✅ SMOTE interpolating one-hot indicators → switched to SMOTENC
- ✅ SMOTE disabled by default (validation experiments showed AUC drop)

**Status:** ✅ Complete and tested

---

### 2024-09-25: Feature Engineering V2

**Implemented:** `src/feature_engineering_v2.py`  
**Author:** Varun  
**Commit:** Part of experiments branch, merged via PR #4 (`a278faf`)

**What It Does:**
- 15 feature groups (external source aggregates, financial burden ratios, age/employment, bureau, previous app, social circle, region risk, property quality, log transforms, missingness indicators)
- Creates 100+ derived features
- Replaces infinities with NaN

**Input:** `data/processed/X_train.csv`, `X_test.csv`  
**Output:** `data/processed/X_train_fe_v2.csv`, `X_test_fe_v2.csv`

**Status:** ✅ Complete  
**Known Issue:** Not importable (module-level execution) → **Fixed Nov 19, 2026** (added `if __name__ == "__main__":` guard)

---

### 2024-09-25: LightGBM + FE V2 with 5-Fold CV

**Implemented:** `src/lgbm_fe_v2_cv.py`  
**Author:** Varun  
**Commit:** `a278faf` (experiments), merged via PR #4

**What It Does:**
- 5-fold stratified CV on FE v2 features
- Early stopping: 150 rounds
- Saves OOF predictions, test predictions, feature importance

**Hyperparameters:**
```python
learning_rate: 0.03, num_leaves: 35, max_depth: 6
min_child_samples: 100, subsample: 0.8, colsample_bytree: 0.8
reg_alpha: 0.5, reg_lambda: 1.0
```

**Results (real data):**

| Fold | Best Iteration | ROC-AUC | PR-AUC | LogLoss |
|---|---|---|---|---|
| 1 | 829 | 0.7710 | 0.2560 | 0.2424 |
| 2 | 903 | 0.7727 | 0.2617 | 0.2415 |
| 3 | 747 | 0.7731 | 0.2617 | 0.2417 |
| 4 | 623 | 0.7647 | 0.2534 | 0.2439 |
| 5 | 860 | 0.7763 | 0.2641 | 0.2407 |
| **Mean** | **792** | **0.7716** | **0.2594** | **0.2420** |

**Status:** ✅ Complete and verified

---

### 2024-09-25: CatBoost + FE V2 with 5-Fold CV

**Implemented:** `src/catboost_fe_v2_cv.py`  
**Author:** Varun  
**Commit:** `a278faf` (experiments), merged via PR #4

**Results (real data):**
- OOF ROC-AUC: 0.7705
- Test ROC-AUC: 0.7700

**Status:** ✅ Complete and verified

---

### 2024-09-25: LightGBM + CatBoost Ensemble

**Implemented:** `src/ensemble_lgbm_catboost.py`  
**Author:** Varun  
**Commit:** `a278faf` (experiments), merged via PR #4

**What It Does:**
- Blends LightGBM and CatBoost OOF predictions
- Grid search over weights (0.00 to 1.00, step 0.05)
- Saves blend results

**Best Blend (real data):**
- Weights: 55% LightGBM, 45% CatBoost
- OOF ROC-AUC: 0.7726
- Improvement over LightGBM alone: +0.0010

**Status:** ✅ Complete and verified

---

### 2024-11-17: Test Set Evaluation

**Implemented:** `src/evaluate_on_test.py`  
**Author:** Varun  
**Status:** Untracked file (needs commit)

**What It Does:**
- Scores LightGBM, CatBoost, and blend on held-out test set
- Loads fold-averaged predictions from CV runs

**Results (real data):**

| Model | Test ROC-AUC | Test PR-AUC | Test LogLoss |
|---|---|---|---|
| **LightGBM FE v2** | **0.7722** | **0.2631** | **0.2417** |
| CatBoost FE v2 | 0.7700 | 0.2599 | 0.2424 |
| Blend (w=0.55) | 0.7720 | 0.2627 | 0.2418 |

**Decision Impact:** LightGBM selected over ensemble (higher test AUC, simpler)

**Status:** ✅ Complete, needs commit

---

### Status: SHAP/LIME Explainability

**Expected:** `src/explainability.py`, global + local SHAP plots, LIME comparison  
**Status:** ⬜ Not started  
**Gap:** Phase 2 deliverable missing (blocks O5 completion)

---

## Phase 1 Kiro Implementation: "Lock the Model"

### 2024-11-19: Model Parameters Extraction

**Implemented:** `src/model_params.py`  
**Author:** Kiro (autonomous)  
**Purpose:** Extract LightGBM hyperparameters from best CV run; define exclusion policy

**Contents:**
- `LGBM_PARAMS`: Hyperparameters from `lgbm_fe_v2_cv.py`
- `EARLY_STOPPING_ROUNDS`: 150
- `EXCLUDED_FEATURES`: ["DAYS_BIRTH"]
- `GENDER_PATTERNS`: Gender exclusion list

**Status:** ✅ Created

---

### 2024-11-19: Three-Band Decision Router

**Implemented:** `src/decision.py`  
**Author:** Kiro (autonomous)  
**Purpose:** Approve / Review / Reject classification logic

**Functions:**
- `decide(probabilities, approve_below, reject_from)`: Batch classification
- `decide_one(probability, approve_below, reject_from)`: Single applicant

**Validation:**
- Threshold ordering (approve < reject)
- Probability range (0-1)
- Edge case handling

**Tests:** `tests/test_decision.py` (7 tests)

**Status:** ✅ Created

---

### 2024-11-19: Final Production Model Training

**Implemented:** `src/final_model.py`  
**Author:** Kiro (autonomous)  
**Purpose:** Train and save production LightGBM with ethical exclusions

**What It Does:**
1. Loads FE v2 features
2. Excludes DAYS_BIRTH and CODE_GENDER_* features
3. 80/20 train/validation split (from training data)
4. Trains LightGBM with early stopping on validation
5. Selects approve/reject thresholds on validation only:
   - Approve: 70% recall target for minority class
   - Reject: 30% precision target, max 10% population share
6. Scores test set once
7. Analyzes decision bands on test set
8. Saves model, model card, band analysis, audit data
9. Optional `--ablation`: Trains with gender, compares performance

**Outputs:**
- `models/final_lgbm.pkl`
- `models/final_model_card.json`
- `reports/final_model_bands.csv`
- `data/processed/audit_test.csv` (for Phase 3 fairness audit)
- `reports/gender_ablation.csv` (if --ablation flag used)

**Tests:** `tests/test_final_model.py` (11 tests)

**Status:** ✅ Created, pending execution

---

### 2024-11-19: Feature Engineering V2 Importability Fix

**Modified:** `src/feature_engineering_v2.py`  
**Author:** Kiro (autonomous)  
**Change:** Added `if __name__ == "__main__":` guard around pipeline execution code

**Rationale:** Allows `add_features()` function to be imported by tests without running the full pipeline

**Tests:** `tests/test_fe_v2.py` (5 tests)

**Status:** ✅ Modified

---

### 2024-11-19: Test Infrastructure

**Implemented:**
- `tests/conftest.py` - Adds `src/` to Python path
- `tests/test_decision.py` - Decision router unit tests
- `tests/test_fe_v2.py` - Feature engineering tests
- `tests/test_final_model.py` - Final model validation tests

**Author:** Kiro (autonomous)

**Test Coverage:**
- Decision logic validation
- Feature engineering determinism
- Gender exclusion verification
- Model card completeness
- Threshold ordering
- AUC floor (0.70 minimum)

**Status:** ✅ Created, pending execution

---

## Change Log

| Date | Phase | Event | Files Changed |
|---|---|---|---|
| 2024-09-25 | 1 | Data preprocessing | `src/data_prep.py` |
| 2024-09-25 | 2 | Baseline models | `src/train_baseline.py` |
| 2024-09-25 | 2 | Scaling pipeline | `src/scale_and_balance.py`, `tests/test_scale_and_balance.py` |
| 2024-09-25 | 2 | Feature Engineering V2 | `src/feature_engineering_v2.py` |
| 2024-09-25 | 2 | LightGBM + FE V2 CV | `src/lgbm_fe_v2_cv.py` |
| 2024-09-25 | 2 | CatBoost + FE V2 CV | `src/catboost_fe_v2_cv.py` |
| 2024-09-25 | 2 | Ensemble experiments | `src/ensemble_lgbm_catboost.py` |
| 2024-11-17 | 2 | Test set evaluation | `src/evaluate_on_test.py` |
| 2024-11-19 | 1 (Kiro) | Model parameters | `src/model_params.py` |
| 2024-11-19 | 1 (Kiro) | Decision router | `src/decision.py` |
| 2024-11-19 | 1 (Kiro) | Final model training | `src/final_model.py` |
| 2024-11-19 | 1 (Kiro) | FE v2 importability fix | `src/feature_engineering_v2.py` |
| 2024-11-19 | 1 (Kiro) | Test infrastructure | `tests/conftest.py`, `tests/test_*.py` |

---

## Pending Work

### Phase 2 Incomplete Items
- ⬜ SHAP global summary plot
- ⬜ SHAP per-applicant waterfall plots
- ⬜ LIME explanations (comparison with SHAP)
- ⬜ Explainability-consistency evaluation (O5 second half)

### Phase 1 (Kiro) Pending
- 🔄 Execute `python3 src/final_model.py --ablation`
- 🔄 Run test suite and verify all pass
- 🔄 Commit Phase 1 files to `dev-varun`

### Uncommitted Files (as of Nov 19, 2026)
- `src/evaluate_on_test.py` (untracked)
- `src/train_lightgbm_recall.py` (untracked, obsolete)
- `src/lgbm_capacity_experiment.py` (untracked, broken)
- Stale reports: `lightgbm_cost_optimized.csv`, `lightgbm_cost_threshold_sweep.csv`, `lightgbm_recall_optimized.csv`
- `backend/models/model.py` (deleted, deletion not staged)

---

**Next Update:** After Phase 1 tests pass and commit is made
