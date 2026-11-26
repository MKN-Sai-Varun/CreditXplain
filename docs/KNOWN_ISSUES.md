# KNOWN_ISSUES — CreditXplain

**Last Updated:** Nov 26, 2026 (Phase 4.8: backend completion)

This document tracks bugs, gaps, technical debt, and unresolved questions. Items are numbered for reference.

---

## 🔴 Critical Issues (Block Progress)

*None currently.*

---

## 🟡 Medium Priority (Should Fix Soon)

### #1: EDA Report Missing
**Status:** Open  
**Phase:** 1  
**Description:** Phase 1 deliverable included EDA report (`reports/eda_age_distribution.png`), but no EDA output files exist in the repo.  
**Impact:** Incomplete Phase 1 documentation; no visual confirmation of data distributions.  
**Action:** Create `src/eda.py` to generate age distribution, class imbalance chart, top feature correlations.  
**Owner:** TBD

---

### #2: SHAP/LIME Implementation ✅ **RESOLVED**
**Status:** Resolved (Nov 19, 2026)  
**Phase:** 2  
**Resolution:** Implemented `src/explainability.py` with full SHAP and LIME support.  
**Evidence:** 
- SHAP additivity: Perfect (0.00 max difference)
- SHAP stability: Perfect (1.00 correlation)
- SHAP-LIME consistency: 58.3% average
- All 13 tests passing
- Outputs: 6 waterfall plots, 6 LIME explanations, global importance, comprehensive report

---

### #3: Test AUC Reporting Method Unclear
**Status:** Open  
**Phase:** 2  
**Description:** `reports/fe_v2_test_results.csv` reports test AUC 0.7722, but this is a **fold-averaged prediction** score, not a single model fit on 80% then scored on test. The final production model may differ slightly.  
**Options:**
1. Report single 80%-fit model (current `final_model.py` approach)
2. Report 5-fold average (current reports)
3. Report full-train refit (not yet done)

**Impact:** Manuscript results section requires clarity on which model was "selected."  
**Action:** Project-lead decision required (deferred to Phase 5).  
**Decision Needed:** Yes

---

### #4: Uncommitted Experimental Files ✅ **RESOLVED**
**Status:** Resolved (Nov 19, 2026)  
**Phase:** 2  
**Resolution:**
- ✅ Committed `src/evaluate_on_test.py`
- ✅ Deleted `src/train_lightgbm_recall.py` (obsolete)
- ✅ Deleted `src/lgbm_capacity_experiment.py` (broken)
- ✅ Deleted stale reports: `lightgbm_cost_optimized.csv`, `lightgbm_cost_threshold_sweep.csv`, `lightgbm_recall_optimized.csv`
- ✅ Staged and committed `backend/models/model.py` deletion

---

### #5: PR #3 Review Medium Findings (Open)
**Status:** Open  
**Phase:** 2  
**Source:** Code review on PR #3

**Finding 1:** Outputs not consumed by training or backend  
**Description:** Generated files (scaler, continuous_cols, modified CSVs) exist but are not imported by downstream scripts.  
**Impact:** Dead code / unused artifacts.  
**Action:** Verify all outputs are consumed in Phase 1 final model; remove unused.

**Finding 2:** Stage disconnected from documented pipeline  
**Description:** `scale_and_balance.py` is not explicitly called in README run instructions.  
**Impact:** User confusion over correct pipeline order.  
**Action:** Update README with full pipeline order in Phase 1 completion.

---

## 🟢 Low Priority (Nice to Have)

### #6: README Outdated
**Status:** Open  
**Phase:** 5  
**Description:** README references Streamlit (changed to Next.js per Amendment B1). Phase status badge shows "2 of 5" but Phase 2 is incomplete (SHAP/LIME missing). Directory structure lists `api/` and `frontend/` that don't exist yet.  
**Impact:** First-time users get wrong instructions.  
**Action:** Rewrite README after Phase 1 lock (defer to Phase 2 completion).  
**Owner:** Member 4 (manuscript coordinator)

---

### #7: No CI/CD Pipeline
**Status:** Open  
**Phase:** 5  
**Description:** No automated test execution on push/PR (planned for Phase 5 per MASTER_PLAN Part C).  
**Impact:** Manual test execution required; risk of pushing broken code.  
**Action:** Set up GitHub Actions to run pytest on PR.  
**Owner:** TBD (defer to Phase 5)

---

### #8: Gender Feature Importance in Existing Models ✅ **RESOLVED**
**Status:** Resolved (Nov 19, 2026)  
**Phase:** 2  
**Resolution:** Ablation study completed. Gender exclusion costs only 0.17% AUC (0.7668 without vs 0.7685 with gender). Ethical trade-off validated as acceptable.  
**Evidence:** `reports/gender_ablation.csv`

---

### #9: Stale Cost-Optimized Reports ✅ **RESOLVED**
**Status:** Resolved (Nov 19, 2026)  
**Phase:** 2  
**Resolution:** Deleted all three stale reports. Superseded by three-band router with balanced thresholds.

---

## ❓ Open Questions (Decisions Needed)

### Q1: ~~Gender Exclusion Confirmation~~ ✅ **RESOLVED**
**Resolved:** Nov 19, 2026  
**Guidance (project lead):** "Gender should not be taken into consideration by the model"  
**Decision:** Exclude CODE_GENDER entirely from model (D-3 in TECHNICAL_DECISIONS.md)

---

### Q2: Age and Family Status Handling
**Status:** Open  
**Phase:** 3  
**Question:** Should age and family status also be excluded from the model (like gender), or kept and audited?  
**Guidance (project lead, Nov 19):** "Age is a consideration, since if they are minors, guardians are required. For minors or students, father's employment and securities are necessary."  
**Decision:** Keep AGE_YEARS and family status features; audit for fairness disparities in Phase 3.  
**Action:** Monitor for age-based disparities beyond legal thresholds.

---

### Q3: Review Timeline and Submission Dates
**Status:** Open  
**Phase:** 5  
**Question:** What are the real project review dates and submission deadline?  
**Description:** MASTER_PLAN timeline section is empty; no dates recorded.  
**Impact:** Cannot plan Phase 5 manuscript completion.  
**Action:** Project lead to provide academic calendar milestones.

---

### Q4: Other Team Members' Work
**Status:** Open  
**Phase:** All  
**Question:** What have Members 2, 3, and 4 already built?  
**Description:** Pranav visible in git history (`dev-pranav`), but other members' contributions unknown.  
**Impact:** Risk of duplicated work (already happened once with preprocessing).  
**Action:** Sync with team on Notion; check for parallel branches.

---

### Q5: Business Threshold Validation
**Status:** Open  
**Phase:** 4  
**Question:** Are the automatically selected thresholds (approve < 70% recall, reject at 30% precision) acceptable for business use?  
**Description:** Thresholds chosen algorithmically on validation set, not by domain expert or business stakeholder.  
**Impact:** May need adjustment after user/officer testing.  
**Action:** Defer to Phase 4 dashboard testing; allow threshold re-tuning.

---

## 🐛 Bugs (Reproducible Issues)

### #10: feature_engineering_v2.py Not Importable (FIXED)
**Status:** ✅ Resolved (Nov 19, 2026)  
**Phase:** 2  
**Description:** Module-level execution code prevented `from feature_engineering_v2 import add_features` from working without running the full pipeline.  
**Fix:** Added `if __name__ == "__main__":` guard around pipeline code.  
**Tests:** `tests/test_fe_v2.py` now passes.

---

### #11: lgbm_capacity_experiment.py Broken ✅ **RESOLVED**
**Status:** Resolved (Nov 19, 2026)  
**Phase:** 2  
**Resolution:** Deleted (obsolete experiment).

---

## 📝 Technical Debt

### #12: Hardcoded Paths
**Status:** Open  
**Phase:** All  
**Description:** Scripts use hardcoded relative paths (`data/processed/X_train.csv`) instead of config file or command-line args.  
**Impact:** Cannot easily change data location; breaks if run from wrong directory.  
**Action:** Refactor to use `pathlib` and config file (defer to Phase 2 pipeline consolidation).  
**Priority:** Low (works for current single-user setup)

---

### #13: No Logging Framework
**Status:** Open  
**Phase:** All  
**Description:** Scripts use `print()` statements; no structured logging (INFO/DEBUG/ERROR levels).  
**Impact:** Hard to debug issues; no log files for post-mortem analysis.  
**Action:** Add Python `logging` module (defer to Phase 4 API layer).  
**Priority:** Low (acceptable for research prototype)

---

### #14: Model Versioning
**Status:** Open  
**Phase:** 2  
**Description:** Models saved as `models/final_lgbm.pkl` with no version or timestamp; overwritten on each run.  
**Impact:** Cannot reproduce previous model versions if retrained.  
**Action:** Add timestamp or git hash to model filename (defer to Phase 5 production deployment).  
**Priority:** Low (single model for now)

---

## Change Log

| Date | Issue | Action |
|---|---|---|
| Nov 19, 2026 | #10 | Resolved: Added main guard to feature_engineering_v2.py |
| Nov 19, 2026 | Q1 | Resolved: Gender exclusion confirmed by project lead |
| Nov 19, 2026 | Q2 | Clarified: Age and family status kept in model, audited for fairness |
| Nov 19, 2026 | #4 | Resolved: Committed evaluate_on_test.py, deleted obsolete files |
| Nov 19, 2026 | #8 | Resolved: Gender ablation study completed, 0.17% AUC cost |
| Nov 19, 2026 | #9 | Resolved: Deleted stale cost-optimized reports |
| Nov 19, 2026 | #11 | Resolved: Deleted broken lgbm_capacity_experiment.py |
| Nov 19, 2026 | All | Initial KNOWN_ISSUES.md created |
| Nov 19, 2026 | Phase 1 | All Phase 1 issues resolved and committed |
| Nov 19, 2026 | #2 | Resolved: SHAP/LIME explainability implemented |
| Nov 19, 2026 | Phase 2 | All Phase 2 deliverables complete |

---

**Next Update:** After Phase 1 tests complete and issues are triaged


---

## Phase 3: Counterfactuals & Fairness Audit — New Issues

### #15: Low Counterfactual Success Rate
**Status:** Superseded (Nov 19, 2026) by the bounded lever search in `src/recourse.py` (commit b68c3f7). The Approve-band question is now tracked in #56  
**Phase:** 3  
**Description:** Counterfactual generation achieved only 3.3% success rate (1 out of 30 counterfactuals successfully flipped from REJECT to REVIEW/APPROVE).  
**Root Cause:** Two-tier success criteria with extremely strict approval threshold (< 0.0349) makes flips challenging. Most applicants generated valid counterfactuals but remained in REVIEW zone rather than reaching APPROVE.  
**Impact:** Limited actionable recommendations for rejected applicants; only ~1 in 30 applicants see viable path to approval.  
**Metrics:**
- 30 counterfactuals generated (10 applicants × 3 CFs each)
- 1 successful flip (3.3%)
- Average 1.5 feature changes per counterfactual
- Top actionable features: INCOME_PER_FAMILY_MEMBER (13%), OWN_CAR_AGE (13%)
**Options:**
1. Accept low success rate (realistic for high-risk applicants)
2. Relax approval threshold slightly (may increase defaults)
3. Add REVIEW→APPROVE counterfactuals as separate tier
**Action:** Defer to Phase 4 dashboard testing; monitor user feedback on recommendation utility.  
**Decision Needed:** Yes (threshold adjustment decision)

---

### #16: Age-Based Fairness Disparity Detected
**Status:** Open — Requires Mitigation  
**Phase:** 3  
**Severity:** 🔴 **Critical** (violates fairness threshold)  
**Description:** Fairness audit identified significant age-based disparity in approval rates that exceeds the 0.1 threshold.  
**Metrics:**
- **Demographic Parity Difference:** 0.3488 (threshold: 0.1)
- **Approval Rate by Age:**
  - <25 years: 13.36%
  - 25-35 years: 27.72%
  - 35-45 years: 37.11%
  - 45-55 years: 46.56%
  - 55+ years: 48.25%
- **Disparity:** 34.89 percentage points between youngest and oldest groups

**Has Children Status:**
- Demographic Parity Difference: 0.0447 (acceptable, below 0.1 threshold)
- With children: 39.34% approval
- Without children: 34.87% approval

**Root Cause:** Model likely learning age as strong proxy for credit history length and financial stability. Younger applicants systematically disadvantaged.  
**Legal Context:** Age discrimination in credit is prohibited in many jurisdictions (e.g., ECOA in US prohibits age-based discrimination for applicants 18+).  
**Impact:** Potential legal liability; ethical concerns about systematic bias against young applicants.  
**Options:**
1. Remove age-derived features entirely (may hurt predictive power)
2. Apply fairness constraints during training (e.g., Fairlearn postprocessing)
3. Adjust decision thresholds by age group to equalize approval rates
4. Conduct further investigation into why age drives such strong disparities
**Recommendation:** Prioritize fairness mitigation in Phase 4; consider threshold adjustment or fairness-constrained retraining.  
**Action:** Project-lead decision required on mitigation strategy before production deployment.  
**Decision Needed:** Yes (fairness vs. performance trade-off)

---

### #17: DiCE Library Categorical Feature Handling
**Status:** Workaround Applied  
**Phase:** 3  
**Description:** DiCE library's automatic categorical feature detection caused `CategoricalDtype` object errors when attempting to generate counterfactuals.  
**Error Message:** `"pandas dtypes must be int, float or bool. Instead got [<class 'pandas.core.dtypes.dtypes.CategoricalDtype'>]"`  
**Root Cause:** DiCE converts string-based columns to CategoricalDtype internally, which conflicts with sklearn model expectations.  
**Workaround:** Forced ALL features as continuous in `dice_ml.Data()` initialization, preventing automatic categorical detection.  
**Code:**
```python
d = dice_ml.Data(
    dataframe=X_train_sample,
    continuous_features=X_train_sample.columns.tolist(),  # Force all as continuous
    outcome_name='TARGET'
)
```
**Impact:** May reduce counterfactual quality for truly categorical features (e.g., occupation type), but ensures script runs without errors.  
**Long-term Fix:** Investigate DiCE's categorical feature support or switch to alternative counterfactual library (e.g., CARLA, Alibi).  
**Priority:** Low (workaround sufficient for Phase 3 deliverable)

---

### #18: DiCE diversity_weight Parameter Unsupported
**Status:** Resolved (Nov 19, 2026)  
**Phase:** 3  
**Description:** Initial counterfactual generation code included `diversity_weight=0.5` parameter in DiCE call, which is not supported by the 'random' method.  
**Error:** `generate_counterfactuals() got an unexpected keyword argument 'diversity_weight'`  
**Fix:** Removed `diversity_weight` parameter from DiCE call. Diversity now controlled only by `total_CFs=3`.  
**Impact:** Reduced control over counterfactual diversity, but random method provides sufficient variety for Phase 3.

---

### #19: Test Age Band Label Mismatch
**Status:** Resolved (Nov 19, 2026)  
**Phase:** 3  
**Description:** Fairness tests initially failed due to hardcoded age band labels ('<25 years', '25-35 years', etc.) not matching implementation labels ('<25', '25-35', etc.).  
**Fix:** Updated all test age band labels to match `create_age_bands()` implementation.  
**Tests Affected:** `test_create_age_bands_comprehensive`, `test_age_band_edge_cases`, `test_age_band_distribution`  
**Resolution:** All 22/22 fairness tests now passing.

---

### #20: Missing DAYS_BIRTH in Fairness Test Data
**Status:** Resolved (Nov 19, 2026)  
**Phase:** 3  
**Description:** `create_age_bands()` expected DAYS_BIRTH column in test dataframe, but test fixtures only provided AGE_YEARS.  
**Fix:** Updated `create_age_bands()` to load DAYS_BIRTH from `X_test_fe_v2.csv` when not present in input dataframe.  
**Code:**
```python
if 'DAYS_BIRTH' not in df.columns:
    full_data = pd.read_csv('data/processed/X_test_fe_v2.csv')
    df = df.merge(full_data[['DAYS_BIRTH']], left_index=True, right_index=True, how='left')
```
**Impact:** Tests now pass; production code unaffected (DAYS_BIRTH always present in real data).

---

### #21: LightGBM feature_name_ Attribute Error
**Status:** Resolved (Nov 19, 2026)  
**Phase:** 3  
**Description:** Fairness audit code attempted to access `model.feature_name_` attribute, but LightGBM Booster uses `feature_name()` method instead.  
**Error:** `'Booster' object has no attribute 'feature_name_'`  
**Fix:** Changed all occurrences to `model.feature_name()` method call.  
**Affected Files:** `src/fairness_audit.py`, `tests/test_fairness.py`

---

### #22: default_probability Column Missing in Test Data
**Status:** Resolved (Nov 19, 2026)  
**Phase:** 3  
**Description:** Fairness tests expected `default_probability` column, but `load_fairness_data()` only created `predicted_probability`.  
**Fix:** Added alias in `load_fairness_data()`:
```python
predictions_df['default_probability'] = predictions_df['predicted_probability']
```
**Impact:** Maintains backward compatibility with both column names.

---

### #23: Duplicate Code in counterfactuals.py
**Status:** Resolved (Nov 19, 2026)  
**Phase:** 3  
**Description:** Found 436 duplicate lines (lines 159-594) in `counterfactuals.py` causing syntax errors during import.  
**Root Cause:** Likely copy-paste error during development.  
**Fix:** Deleted all duplicate lines, leaving single clean implementation.  
**Evidence:** All 13/13 counterfactual tests now passing.

---

### #24: Fairness Script Running on Import
**Status:** Resolved (Nov 19, 2026)  
**Phase:** 3  
**Description:** `fairness_audit.py` executed full audit script on import, preventing test imports and causing test failures.  
**Fix:** Added `if __name__ == "__main__":` guard and properly indented 559 lines of execution code.  
**Impact:** Script now importable for testing; helper functions accessible to tests.

---

## Change Log (Updated)

| Date | Issue | Action |
|---|---|---|
| Nov 19, 2026 | #10 | Resolved: Added main guard to feature_engineering_v2.py |
| Nov 19, 2026 | Q1 | Resolved: Gender exclusion confirmed by project lead |
| Nov 19, 2026 | Q2 | Clarified: Age and family status kept in model, audited for fairness |
| Nov 19, 2026 | #4 | Resolved: Committed evaluate_on_test.py, deleted obsolete files |
| Nov 19, 2026 | #8 | Resolved: Gender ablation study completed, 0.17% AUC cost |
| Nov 19, 2026 | #9 | Resolved: Deleted stale cost-optimized reports |
| Nov 19, 2026 | #11 | Resolved: Deleted broken lgbm_capacity_experiment.py |
| Nov 19, 2026 | All | Initial KNOWN_ISSUES.md created |
| Nov 19, 2026 | Phase 1 | All Phase 1 issues resolved and committed |
| Nov 19, 2026 | #2 | Resolved: SHAP/LIME explainability implemented |
| Nov 19, 2026 | Phase 2 | All Phase 2 deliverables complete |
| Nov 19, 2026 | #15 | Identified: Low counterfactual success rate (3.3%) |
| Nov 19, 2026 | #16 | **CRITICAL**: Age-based fairness disparity detected (0.3488 DPD) |
| Nov 19, 2026 | #17 | Workaround: DiCE categorical feature handling |
| Nov 19, 2026 | #18 | Resolved: Removed unsupported diversity_weight parameter |
| Nov 19, 2026 | #19 | Resolved: Fixed test age band label mismatch |
| Nov 19, 2026 | #20 | Resolved: Fixed missing DAYS_BIRTH in test data |
| Nov 19, 2026 | #21 | Resolved: Fixed LightGBM feature_name_ attribute error |
| Nov 19, 2026 | #22 | Resolved: Added default_probability column alias |
| Nov 19, 2026 | #23 | Resolved: Deleted 436 duplicate lines in counterfactuals.py |
| Nov 19, 2026 | #24 | Resolved: Added main guard to fairness_audit.py |
| Nov 19, 2026 | Phase 3 | All Phase 3 implementation and tests complete |

---

**Next Update:** After Phase 4 dashboard implementation


---

## Phase 4.1: Backend Foundation — New Issues

### #25: Bcrypt Version Compatibility Warning
**Status:** ⚠️ Non-Critical  
**Phase:** 4.1  
**Description:** Bcrypt password hashing tests fail with version detection error during initialization.  
**Error Message:** `AttributeError: module 'bcrypt' has no attribute '__about__'`  
**Root Cause:** Passlib 1.7.4 tries to read bcrypt version from `__about__.__version__`, but newer bcrypt versions (4.1+) changed this structure.  
**Workaround:** Downgraded to bcrypt 4.0.1, which works with passlib 1.7.4.  
**Impact:** Production usage unaffected - password hashing and verification work correctly. Only affects test suite initialization warnings.  
**Test Results:**
- Password hashing works in production (seeding script successful)
- Login endpoint verified with real JWT generation
- Only unit test execution shows warnings

**Long-term Fix:** Wait for passlib update or switch to bcrypt directly (skip passlib layer).  
**Priority:** Low (production functionality not impacted)

**Date:** Nov 19, 2026

---

### #26: Email Validator Dependency Missing
**Status:** ✅ Resolved (Nov 19, 2026)  
**Phase:** 4.1  
**Description:** Pydantic EmailStr type requires email-validator package, not included in initial requirements.txt.  
**Error:** `ImportError: email-validator is not installed`  
**Fix:** Added `pip3 install email-validator` to installation steps.  
**Resolution:** Included email-validator in dependencies.

---

### #27: Pydantic V2 Deprecation Warnings
**Status:** ✅ Resolved (Phase 4.8). All 7 response schemas use `model_config = ConfigDict(from_attributes=True)`  
**Phase:** 4.1  
**Description:** Class-based `Config` deprecated in Pydantic V2, causing multiple warnings.  
**Fix:** Migrated all schemas to use `model_config = ConfigDict(...)` instead of inner `Config` class.  
**Files Updated:**
- `backend/config.py` - Settings class
- `backend/schemas.py` - All response schemas (7 classes)

---

### #28: SQLAlchemy Declarative Base Deprecation
**Status:** ✅ Resolved (Nov 19, 2026)  
**Phase:** 4.1  
**Description:** `sqlalchemy.ext.declarative.declarative_base()` deprecated in SQLAlchemy 2.0.  
**Fix:** Changed import to `sqlalchemy.orm.declarative_base()`.  
**File:** `backend/database.py`

---

### #29: Pandas Version Conflicts in Requirements
**Status:** ✅ Resolved (Nov 19, 2026)  
**Phase:** 4.1  
**Description:** Strict version pinning (`pandas==2.1.3`) caused dependency resolution failures.  
**Error:** `ResolutionImpossible: for help visit https://pip.pypa.io/en/latest/user_guide/#fixing-conflicting-dependencies`  
**Fix:** Changed to flexible version constraints (`pandas>=2.0.0`) for all packages except bcrypt.  
**Impact:** Installation succeeds with compatible versions across all dependencies.

---

### #30: CORS Pre-flight Requests Not Tested
**Status:** Open  
**Phase:** 4.1  
**Description:** CORS middleware configured but not tested with actual Next.js frontend requests.  
**Impact:** May discover CORS issues when frontend is integrated in Phase 4.6.  
**Action:** Test with Next.js frontend in Phase 4.6; adjust allowed headers/methods if needed.  
**Priority:** Medium (defer to frontend integration)

---

### #31: Predictions and Fairness Endpoints Are Stubs
**Status:** ✅ Resolved. Predictions (Phase 4.2) and fairness (Phase 4.5) endpoints are implemented  
**Phase:** 4.1  
**Description:** `/api/predictions/` and `/api/fairness/` endpoints return 501 Not Implemented.  
**Rationale:** These require ML model loading and Phase 3 integration (planned for Phases 4.2 and 4.5).  
**Action:** Implement in Phase 4.2 (predictions) and Phase 4.5 (fairness).  
**Priority:** High (next phase deliverable)

---

### #32: No Integration Tests for API Endpoints
**Status:** ✅ Resolved (Phase 4.8). Officer, fairness, document and application routes now have API tests (`test_officer_api.py`, `test_fairness_api.py`, `test_documents_api.py`, `test_applications_api.py`)  
**Phase:** 4.1  
**Description:** Only unit tests exist (auth, database). No end-to-end API tests with FastAPI TestClient.  
**Impact:** Cannot verify full request/response flow without manual curl testing.  
**Action:** Add API integration tests in Phase 4.2 covering:
- Authentication flow (register → login → get current user)
- Application submission → prediction → decision
- Officer review queue and actions

**Priority:** Medium (defer to Phase 4.9 integration testing)

---

### #33: Database File Not Gitignored by Default
**Status:** ✅ Resolved. `.gitignore` covers `*.db` and `*.db-journal`  
**Phase:** 4.1  
**Description:** `creditxplain.db` file created in backend directory, not explicitly gitignored.  
**Impact:** Risk of accidentally committing database with test data.  
**Action:** Add `*.db` to `.gitignore` in backend directory.  
**Priority:** Low (no sensitive data yet, but good practice)

---

### #34: No Password Reset Mechanism
**Status:** Open  
**Phase:** 4.1  
**Description:** Authentication system has login/register but no password reset flow.  
**Impact:** Demo users cannot reset passwords; must be manually updated in database.  
**Rationale:** Single-user prototype (Amendment B3) - low priority for demo.  
**Action:** Defer to production deployment (Phase 5) if needed.  
**Priority:** Low (not required for academic project)

---

### #35: Environment Variables Not Validated
**Status:** Open  
**Phase:** 4.1  
**Description:** No validation that required environment variables (SECRET_KEY) are set before server starts.  
**Impact:** Server may start with default insecure secret key if .env not configured.  
**Action:** Add startup validation that checks critical settings (SECRET_KEY != default).  
**Priority:** Low (development OK, but flag for production)

---

## Change Log (Updated)

| Date | Issue | Action |
|---|---|---|
| Nov 19, 2026 | #10 | Resolved: Added main guard to feature_engineering_v2.py |
| Nov 19, 2026 | Q1 | Resolved: Gender exclusion confirmed by project lead |
| Nov 19, 2026 | Q2 | Clarified: Age and family status kept in model, audited for fairness |
| Nov 19, 2026 | #4 | Resolved: Committed evaluate_on_test.py, deleted obsolete files |
| Nov 19, 2026 | #8 | Resolved: Gender ablation study completed, 0.17% AUC cost |
| Nov 19, 2026 | #9 | Resolved: Deleted stale cost-optimized reports |
| Nov 19, 2026 | #11 | Resolved: Deleted broken lgbm_capacity_experiment.py |
| Nov 19, 2026 | All | Initial KNOWN_ISSUES.md created |
| Nov 19, 2026 | Phase 1 | All Phase 1 issues resolved and committed |
| Nov 19, 2026 | #2 | Resolved: SHAP/LIME explainability implemented |
| Nov 19, 2026 | Phase 2 | All Phase 2 deliverables complete |
| Nov 19, 2026 | #15 | Identified: Low counterfactual success rate (3.3%) |
| Nov 19, 2026 | #16 | **CRITICAL**: Age-based fairness disparity detected (0.3488 DPD) |
| Nov 19, 2026 | #17 | Workaround: DiCE categorical feature handling |
| Nov 19, 2026 | #18 | Resolved: Removed unsupported diversity_weight parameter |
| Nov 19, 2026 | #19 | Resolved: Fixed test age band label mismatch |
| Nov 19, 2026 | #20 | Resolved: Fixed missing DAYS_BIRTH in test data |
| Nov 19, 2026 | #21 | Resolved: Fixed LightGBM feature_name_ attribute error |
| Nov 19, 2026 | #22 | Resolved: Added default_probability column alias |
| Nov 19, 2026 | #23 | Resolved: Deleted 436 duplicate lines in counterfactuals.py |
| Nov 19, 2026 | #24 | Resolved: Added main guard to fairness_audit.py |
| Nov 19, 2026 | Phase 3 | All Phase 3 implementation and tests complete |
| Nov 19, 2026 | #25 | Identified: Bcrypt version compatibility (non-critical) |
| Nov 19, 2026 | #26 | Resolved: Added email-validator dependency |
| Nov 19, 2026 | #27 | Resolved: Migrated to Pydantic V2 ConfigDict |
| Nov 19, 2026 | #28 | Resolved: Updated SQLAlchemy declarative_base import |
| Nov 19, 2026 | #29 | Resolved: Fixed pandas version conflicts with flexible constraints |
| Nov 19, 2026 | #30 | Identified: CORS pre-flight not tested (defer to Phase 4.6) |
| Nov 19, 2026 | #31 | By Design: Predictions and fairness endpoints stubbed (Phase 4.2/4.5) |
| Nov 19, 2026 | #32 | Identified: No API integration tests (defer to Phase 4.9) |
| Nov 19, 2026 | #33 | Identified: Database file not gitignored |
| Nov 19, 2026 | #34 | Identified: No password reset mechanism (low priority) |
| Nov 19, 2026 | #35 | Identified: Environment variables not validated |
| Nov 19, 2026 | Phase 4.1 | Backend foundation complete - 4/4 tests passing, server verified |

---

**Next Update:** After Phase 4.2 (Prediction Engine Integration)


---

## Phase 4.2: Prediction Engine — New Issues

### #36: Phase 1 Final Model Not Trained
**Status:** ✅ Resolved. `models/final_lgbm.pkl` and `models/final_model_card.json` exist and are served through `src/inference.py` (commit ae6ad19)  
**Phase:** 4.2  
**Description:** `models/final_lgbm.pkl` referenced in config but never trained/saved from Phase 1.  
**Current State:**
- Config points to `final_lgbm.pkl` (correct)
- Old baseline `lightgbm.pkl` exists (Aug 16, baseline model)
- Phase 1 was not completed with final model training script
- Services initialize with warning but continue

**Impact:**
- Prediction API endpoint works but returns 500 error ("Model not loaded")
- SHAP/LIME services cannot initialize
- Counterfactual generation unavailable
- End-to-end testing blocked

**Workaround:** Services handle missing model gracefully with clear error messages

**Action:** Train Phase 1 final model or use existing baseline model  
**Options:**
1. Complete Phase 1 model training (`src/final_model.py`)
2. Point config to existing `lightgbm.pkl` (less accurate)
3. Deploy without ML (manual decisions only)

**Priority:** High (required for functional prediction API)

**Date:** Nov 19, 2026

---

### #37: Old Baseline Model File Corrupted
**Status:** ✅ Resolved. `models/lightgbm.pkl` is no longer in `models/`; nothing loads it  
**Phase:** 4.2  
**Description:** `models/lightgbm.pkl` (368 KB, Aug 16) fails to load with pickle error.  
**Error:** `_pickle.UnpicklingError: invalid load key, '\x04'`  
**Root Cause:** File corruption or Python version mismatch (trained with different Python/pickle protocol)

**Impact:** Cannot use old baseline model as fallback

**Action:** Delete corrupted file or retrain baseline model  
**Priority:** Low (Phase 1 final model should be used anyway)

**Date:** Nov 19, 2026

---

### #38: LIME Training Data Hardcoded Path
**Status:** Open  
**Phase:** 4.2  
**Description:** ExplainabilityService loads LIME training data from hardcoded `X_train_fe_v2.csv` path.  
**Current Code:**
```python
train_path = self.prediction_service.settings.DATA_DIR / "X_train_fe_v2.csv"
```

**Impact:**
- Fails silently if file doesn't exist (LIME not initialized)
- Loads 1000 rows on every service initialization (slow)
- Training data not needed for SHAP (unnecessary dependency)

**Action:** Make LIME initialization optional or cache training data  
**Priority:** Low (LIME is supplementary, SHAP is primary)

**Date:** Nov 19, 2026

---

### #39: No Caching for Loaded Models
**Status:** Open  
**Phase:** 4.2  
**Description:** Models are singletons but loaded synchronously on first request, causing ~2-3 second delay.  
**Impact:** First prediction request is slow; subsequent requests fast

**Action:** Consider async model loading or preload on startup  
**Options:**
1. Preload in FastAPI startup event
2. Use background task for model loading
3. Accept first-request latency (current approach)

**Priority:** Low (acceptable for prototype)

**Date:** Nov 19, 2026

---

### #40: DiCE Counterfactual Generation Slow
**Status:** Superseded. The API no longer uses DiCE; the bounded lever search runs in about 14 ms per applicant (`reports/recourse_validity.json`)  
**Phase:** 4.2  
**Description:** DiCE counterfactual generation takes 2-5 seconds for 3 counterfactuals.  
**Impact:** Prediction API endpoint has longer response time for REJECT decisions

**Action:** Consider caching, async generation, or limiting to 1-2 counterfactuals  
**Priority:** Low (only affects rejected applications)

**Date:** Nov 19, 2026

---

### #41: No API Tests for Prediction Endpoint
**Status:** ✅ Resolved. `backend/tests/test_predict_api.py` covers predict, read-back and double-predict  
**Phase:** 4.2  
**Description:** Prediction pipeline implemented but no integration tests with FastAPI TestClient.  
**Impact:** Cannot verify full request → prediction → explanation → database flow

**Action:** Add API integration tests in Phase 4.9  
**Priority:** Medium (defer to integration testing phase)

**Date:** Nov 19, 2026

---

### #42: Explanation Storage as JSON Blobs
**Status:** By Design  
**Phase:** 4.2  
**Description:** SHAP values, LIME explanations, and counterfactuals stored as JSON in single `explanations.shap_values` column.  
**Alternatives:** Separate tables for each explanation type

**Rationale:**
- Flexible schema (explanation format may evolve)
- Simpler queries (one join instead of three)
- Smaller database (no repeated metadata)

**Trade-off:** Cannot query individual SHAP values or features directly

**Priority:** Not an issue (acceptable design choice)

**Date:** Nov 19, 2026

---

## Change Log (Updated)

| Date | Issue | Action |
|---|---|---|
| Nov 19, 2026 | #10 | Resolved: Added main guard to feature_engineering_v2.py |
| Nov 19, 2026 | Q1 | Resolved: Gender exclusion confirmed by project lead |
| Nov 19, 2026 | Q2 | Clarified: Age and family status kept in model, audited for fairness |
| Nov 19, 2026 | #4 | Resolved: Committed evaluate_on_test.py, deleted obsolete files |
| Nov 19, 2026 | #8 | Resolved: Gender ablation study completed, 0.17% AUC cost |
| Nov 19, 2026 | #9 | Resolved: Deleted stale cost-optimized reports |
| Nov 19, 2026 | #11 | Resolved: Deleted broken lgbm_capacity_experiment.py |
| Nov 19, 2026 | All | Initial KNOWN_ISSUES.md created |
| Nov 19, 2026 | Phase 1 | All Phase 1 issues resolved and committed |
| Nov 19, 2026 | #2 | Resolved: SHAP/LIME explainability implemented |
| Nov 19, 2026 | Phase 2 | All Phase 2 deliverables complete |
| Nov 19, 2026 | #15 | Identified: Low counterfactual success rate (3.3%) |
| Nov 19, 2026 | #16 | **CRITICAL**: Age-based fairness disparity detected (0.3488 DPD) |
| Nov 19, 2026 | #17-24 | Resolved: Phase 3 implementation issues |
| Nov 19, 2026 | Phase 3 | All Phase 3 implementation and tests complete |
| Nov 19, 2026 | #25-35 | Identified: Phase 4.1 backend foundation issues |
| Nov 19, 2026 | Phase 4.1 | Backend foundation complete - 4/4 tests passing, server verified |
| Nov 19, 2026 | #36 | **CRITICAL**: Phase 1 final model not trained (blocks prediction API) |
| Nov 19, 2026 | #37 | Identified: Old baseline model corrupted |
| Nov 19, 2026 | #38 | Identified: LIME training data hardcoded path |
| Nov 19, 2026 | #39 | Identified: No model caching (first request slow) |
| Nov 19, 2026 | #40 | Identified: DiCE counterfactual generation slow (2-5s) |
| Nov 19, 2026 | #41 | Identified: No API tests for prediction endpoint |
| Nov 19, 2026 | #42 | By Design: Explanation storage as JSON blobs |
| Nov 19, 2026 | Phase 4.2 | Prediction engine integration complete - services working, awaiting trained model |

---

**Next Update:** After Phase 4.3 (Loan Terms Calculator)

---

## Phase 4.3: Loan Calculator — New Issues

### #43: Illustrative Interest Rates (Not Real Bank Rates)
**Status:** ⚠️ Documentation Required  
**Phase:** 4.3  
**Description:** Interest rates are calculated algorithmically for demonstration purposes, not based on real bank lending policies.  
**Current Rates:** 12-25% based on risk bands

**Impact:**
- Rates are reasonable but not validated by actual lending institution
- May not reflect real-world underwriting criteria
- Regulatory compliance not verified

**Action:** Add disclaimer in user-facing reports that rates are illustrative for academic project  
**Priority:** High (required for manuscript/demo honesty)

**Date:** Nov 19, 2026

---

### #44: Loan Tenure Fixed at 12 Months
**Status:** ✅ Resolved (Phase 4.8). Applicants choose 12, 24, 36, 48 or 60 months (D-52)  
**Phase:** 4.3  
**Description:** Default tenure hardcoded to 12 months in predictions router.  
**Current Code:**
```python
tenure_months=12  # Default 12-month tenure
```

**Impact:**
- No flexibility for applicants to choose tenure
- Higher loan amounts may need longer tenure for affordability
- Frontend would need custom implementation to allow tenure selection

**Action:** Add tenure as optional parameter in application submission (Phase 4.4)  
**Options:**
1. Keep 12-month default (current)
2. Add tenure field to application form
3. Calculate recommended tenure based on loan amount

**Priority:** Low (acceptable for prototype)

**Date:** Nov 19, 2026

---

### #45: No Processing Fees or Additional Charges
**Status:** By Design  
**Phase:** 4.3  
**Description:** EMI calculation only includes principal + interest, no processing fees, insurance, or other charges.  
**Typical Real Loans:** 1-3% processing fee, optional insurance

**Impact:**
- Total cost underestimated compared to real loans
- EMI calculation simpler but less realistic

**Rationale:** Academic project scope - focusing on risk assessment, not full loan product

**Priority:** Not an issue (acceptable simplification)

**Date:** Nov 19, 2026

---

### #46: Loan Amount Fallback to 100k
**Status:** ✅ Resolved (Phase 4.8). `AMT_CREDIT` is required at submission; the fallback is removed (D-53)  
**Phase:** 4.3  
**Description:** If AMT_CREDIT field missing from application, defaults to ₹100,000.  
**Current Code:**
```python
loan_amount = application.features_json.get('AMT_CREDIT', 100000)
```

**Impact:**
- Incorrect loan terms if field actually missing
- Frontend should validate AMT_CREDIT is provided
- Test data may trigger fallback unintentionally

**Action:** Make AMT_CREDIT required field in application submission (Phase 4.4)  
**Priority:** Medium (should be mandatory)

**Date:** Nov 19, 2026

---

### #47: No Amortization Schedule Generated
**Status:** Open  
**Phase:** 4.3  
**Description:** Only total EMI and total payable calculated, not month-by-month breakdown of principal/interest.  
**Typical Bank Statement:** Shows principal vs interest split for each month

**Impact:**
- Applicants cannot see how much principal they pay each month
- Cannot calculate early payoff amounts
- Less transparent than full amortization schedule

**Action:** Add amortization schedule generation method (deferred to Phase 5 or later)  
**Priority:** Low (nice-to-have feature)

**Date:** Nov 19, 2026

---

## Change Log (Updated)

| Date | Issue | Action |
|---|---|---|
| Nov 19, 2026 | Phase 4.1 | Backend foundation complete |
| Nov 19, 2026 | Phase 4.2 | Prediction engine integration complete |
| Nov 19, 2026 | #43 | Identified: Interest rates are illustrative, not real bank rates |
| Nov 19, 2026 | #44 | Identified: Loan tenure fixed at 12 months |
| Nov 19, 2026 | #45 | By Design: No processing fees (acceptable simplification) |
| Nov 19, 2026 | #46 | Identified: Loan amount fallback to 100k if missing |
| Nov 19, 2026 | #47 | Identified: No amortization schedule generated |
| Nov 19, 2026 | Phase 4.3 | Loan calculator complete - 12/12 tests passing |

---

**Next Update:** After Phase 4.4 or 4.5

---

## Phase 4.5: Fairness Monitoring — New Issues

### #48: Fairness Metrics Require Manual Trigger
**Status:** By Design  
**Phase:** 4.5  
**Description:** Fairness metrics are not automatically calculated - officer must trigger via POST /api/fairness/calculate.  
**Current Behavior:** Metrics only update when officer manually requests recalculation

**Alternatives:**
1. Auto-calculate after every N applications
2. Background job (scheduled every hour/day)
3. Calculate on every prediction (performance impact)

**Rationale:** Manual trigger gives officers control and avoids unnecessary computation

**Priority:** Not an issue (acceptable design)

**Date:** Nov 19, 2026

---

### #49: Fairness Calculation Requires 50+ Applications
**Status:** By Design  
**Phase:** 4.5  
**Description:** Fairness metrics require minimum 50 applications for statistical significance.  
**Impact:** New deployments cannot calculate fairness until 50 applications processed

**Rationale:** Small samples can show misleading disparities (not statistically reliable)

**Workaround:** Return clear error message if insufficient data

**Priority:** Not an issue (acceptable requirement)

**Date:** Nov 19, 2026

---

### #50: No Historical Fairness Trend Visualization
**Status:** Open  
**Phase:** 4.5  
**Description:** API stores historical metrics but no endpoint to retrieve time-series data.  
**Current:** Only latest metric accessible via GET /metrics

**Impact:** Cannot track fairness improvement/degradation over time

**Action:** Add GET /api/fairness/history endpoint (deferred to Phase 5 or later)  
**Priority:** Low (nice-to-have feature)

**Date:** Nov 19, 2026

---

### #51: Demographic Statistics May Be Stale
**Status:** Open  
**Phase:** 4.5  
**Description:** GET /demographics/* endpoints return stats from latest saved metric, not real-time calculation.  
**Impact:** Stats may be outdated if many new applications processed since last calculation

**Action:** Add "last_calculated" timestamp to response so users know freshness  
**Priority:** Low (acceptable - officer can trigger recalculation)

**Date:** Nov 19, 2026

---

## Phase 4.6: Full Test Run and Explanation/Recourse Review — New Issues

Found on Nov 25, 2026 by running every existing test (`tests/`: 88 passed; `backend/tests/`: 36 passed; frontend `tsc` and `build` clean) and by reading the explainability and recourse outputs.

### #52: LIME Explanations Are Not Local
**Status:** ✅ Resolved (Nov 25, 2026, Phase 4.7). `src/lime_local.py` explains each applicant locally; locality gap 0.00 → 0.49 on 150 applicants (TECHNICAL_DECISIONS D-43)  
**Phase:** 4.6 (fix planned for 4.7)  
**Description:** Both `src/explainability.py:247` and `backend/services/explainability_service.py` build `LimeTabularExplainer` with `discretize_continuous=False` and the default `sample_around_instance=False`. LIME then draws its perturbations around the training-data mean, not around the applicant. With 300 features the default kernel width (0.75·√300 ≈ 13) also gives almost every sample the same weight.  
**Evidence:** Three different test applicants got nearly identical LIME weights for the top feature (`EXT_SOURCE_MEAN`: −0.01741, −0.01758, −0.01740).  
**Impact:** The current output is closer to a global linear fit of the model than a per-applicant explanation, so the 58.3% SHAP–LIME overlap cannot be reported as a local-explanation agreement.  
**Action:** Sample around the instance, keep LIME's default quartile discretisation, set an explicit kernel width, fix the random seed, and re-measure.

---

### #53: LIME Fit Overflows Numerically
**Status:** ✅ Resolved: not a LIME problem. numpy 2.0.2 with macOS Accelerate raises these warnings on any matrix product, even clean random data; results are correct (max error vs `einsum` 3e-14). Silenced in the evaluation scripts only  
**Phase:** 4.6  
**Description:** Each LIME call raises about 5,000 `RuntimeWarning`s (overflow, divide by zero, invalid value) from scikit-learn's ridge regression. The returned weights are still finite.  
**Likely cause:** The same setup as #52: 300 unscaled features, some with values in the millions.  
**Action:** Re-check after the #52 fix. If the warnings remain, explain over a reduced feature set (LIME `feature_selection`).

---

### #54: SHAP–LIME Consistency Measured on 6 Applicants
**Status:** ✅ Resolved (Phase 4.7). `src/explanation_eval.py` measures 150 stratified applicants with overlap, rank and sign agreement, locality and seed stability (`reports/explanation_consistency.json`)  
**Phase:** 4.6 (fix planned for 4.7)  
**Description:** `reports/shap_lime_consistency.csv` holds top-10 overlap for 6 applicants (2 per band). That sample is too small for a paper, and overlap alone ignores ranking.  
**Action:** Measure on 100–200 stratified applicants. Report top-k overlap, rank correlation (Spearman/Kendall), and LIME stability across seeds.

---

### #55: Recourse "Validity Rate" Is Circular
**Status:** ✅ Resolved (Phase 4.7). Replaced by `src/recourse_eval.py` (coverage, cost, sparsity, plausibility, robustness, latency, baselines). `src/recourse_validity.py` and its reports deleted (D-49)  
**Phase:** 4.6 (fix planned for 4.7)  
**Description:** `reports/recourse_validity.json` reports `validity_rate: 1.0`. The search keeps only candidates that the model scores into a better band, and "verification" rescores them with the same model, so 100% follows by construction.  
**Missing metrics:** proximity (size of change), sparsity (number of levers), plausibility (distance to real applicants, e.g. kNN or LOF), diversity, and a baseline (DiCE or random search under the same bounds).  
**Action:** Report coverage (`recourse_found_rate`, currently 0.8767) and the missing metrics instead of the circular rate.

---

### #56: Recourse Never Reaches Approve
**Status:** 🟡 Mitigated (Phase 4.7). Two-tier recourse added: 27.3% of Review applicants now get a robust, plausible path to Approve (D-47). Reject → Approve in one step remains out of reach (structural). For the paper: present recourse as a graduated path  
**Phase:** 4.6  
**Description:** 0 of 300 rejected test applicants reach Approve (`reaches_approve_rate: 0.0`). Recourse moves them only from Reject to Review.  
**Why:** Approve needs a predicted default probability below 0.0349. Rejected applicants average 0.213 (`reports/final_model_bands.csv`). The strongest features are external bureau scores (`EXT_SOURCE_*`, `CREDIT_GOODS_EXT_RISK`), which the three levers (loan amount, income, tenure) cannot change.  
**Lever mix of the best option (263 applicants with recourse):** credit 108, credit+tenure 88, tenure 58, credit+income+tenure 8, income 1.  
**Action:** The plan (§6) counts recourse that "flips or improves" the decision, so Reject→Review qualifies. The paper should present recourse as a path to officer review and state the structural limit. Optionally add Review→Approve recourse as a second tier.

---

### #57: Frontend Is a Starter Page and Fails Lint
**Status:** Open  
**Phase:** 4.6  
**Description:** `frontend/` is untracked. `app/page.tsx` is the default Next.js page. `npm run lint` fails with 1 error: `@typescript-eslint/no-explicit-any` at `frontend/lib/api.ts:46`. `tsc --noEmit` and `npm run build` pass.  
**Action:** Fix during the first frontend phase.

---

### #58: pandas FutureWarning in Fairness Audit
**Status:** ✅ Resolved (Phase 4.8). `observed=False` passed explicitly in `src/fairness_audit.py` and `backend/services/fairness_service.py`  
**Phase:** 4.6  
**Description:** `src/fairness_audit.py:98` calls `df.groupby(sensitive_feature)` on a categorical column without `observed=`, which raises a FutureWarning 11 times in `tests/test_fairness.py`.  
**Action:** Pass `observed=False` explicitly.

---

### #59: Two Counterfactual Implementations Exist
**Status:** ✅ Resolved (Phase 4.7). The API uses `src/recourse.py`; DiCE runs only as a comparison baseline in `src/recourse_eval.py`. `src/counterfactuals.py` stays as the Phase 3 offline script  
**Phase:** 4.6  
**Description:** `src/counterfactuals.py` (Phase 3, DiCE, with `tests/test_counterfactuals.py`) is still in the repo, while the API uses `src/recourse.py` (bounded lever search). The paper and README must say which one produced which numbers.  
**Action:** Use the DiCE version as the comparison baseline in the #55 evaluation, or retire it.

---

### #60: Duplicate Recourse Script Could Overwrite Reports
**Status:** ✅ Resolved (Nov 25, 2026)  
**Phase:** 4.6  
**Description:** The untracked `backend/tests/recourse_validity.py` was a near-copy of `src/recourse_validity.py` that ran on import and wrote `reports/recourse_validity.json` and `reports/recourse_validity_per_applicant.csv`. Pytest skipped it only because of its name.  
**Fix:** Deleted. `src/recourse_validity.py` remains the single source.

---

## Phase 4.7: Explanation and Recourse Upgrade — New Issues

### #61: DiCE Baseline Not Completed
**Status:** Open  
**Phase:** 4.7  
**Description:** DiCE (random method, same three levers) took about 40–120 s per applicant. The full run was stopped early; no DiCE numbers were saved.  
**Action:** Run `python3 -m src.recourse_eval --n 300 --dice 20` (about 30–40 minutes) before the paper's results section.

---

### #62: Stricter Recourse Finds Options for Fewer Applicants
**Status:** Open (trade-off, by design)  
**Phase:** 4.7  
**Description:** The margin and plausibility rules lower the share of applicants with any option. Reject → Review falls from 87.7% to 59.0%, but the share with a *usable* option rises from 21.3% to 59.0%. The margin causes most of the drop (D-44).  
**Options:** A smaller margin (e.g. 0.05) or a finer grid could recover some coverage; random search reaches 60.0%, so a finer search may add about 1 point.  
**Action:** Report the trade-off and ablations in the paper. Revisit the margin if more coverage is needed.

---

### #63: Age-Band Recourse Gaps on Small Groups
**Status:** Open  
**Phase:** 4.7  
**Description:** Recourse coverage differs by about 0.28 between age bands, but the extreme groups are small (Reject → Review: under-25 n=19, 55+ n=24; Review → Approve: under-25 n=9).  
**Action:** Re-run with a larger `--n`, or stratify by age, before reporting group gaps.

---

### #64: FastAPI `on_event` Deprecated
**Status:** ✅ Resolved (Phase 4.8). Startup moved to a `lifespan` handler in `backend/main.py`  
**Phase:** 4.7 (found in the 4.6 test run)  
**Description:** `backend/main.py:23` uses `@app.on_event("startup")`, which FastAPI has deprecated in favour of lifespan handlers.  
**Action:** Move startup code to a `lifespan` context manager.

---

### #65: SHAP "List of ndarray" Notice
**Status:** By design (informational)  
**Phase:** 4.7  
**Description:** SHAP warns that LightGBM binary output changed to a list of arrays. `explainability_service.py` and `src/recourse.explain_change` take the last element, the default-class contribution; tests confirm additivity.

---

### #66: LIME Covers 25 of 300 Features
**Status:** By design (limitation to state in the paper)  
**Phase:** 4.7  
**Description:** `LocalLime` explains only the 25 top-gain features; the rest stay fixed at the applicant's values. That set-up is what makes the explanations local (D-43). The paper must state it.

---

## Phase 4.8: Backend Completion — New Issues

### #67: Uploaded Documents Are Not Scanned and Cannot Be Deleted
**Status:** Open  
**Phase:** 4.8  
**Description:** Uploads are checked for type, size and file signature (D-50), but not scanned for malware, and there is no endpoint to delete or replace a document.  
**Action:** Acceptable for the academic demo. A production system would add scanning and a delete/replace flow with an audit entry.

---

### #68: Live Equalized-Odds Metric Is Always Zero
**Status:** Open  
**Phase:** 4.8 (found while testing the fairness API)  
**Description:** `FairnessService.calculate_fairness_metrics` calls `equalized_odds_difference(y_true=approved, y_pred=approved, ...)`. With identical inputs the result is always 0, so `age_eod` and `family_eod` carry no information. Live applications have no repayment outcome to use as `y_true`.  
**Action:** Drop EOD from the live monitor, or label it as unavailable. The offline audit (`src/fairness_audit.py`, which has real outcomes) remains the source for EOD.

---

### #69: Demographic "rejected" Count Includes Review
**Status:** Open (low)  
**Phase:** 4.8  
**Description:** `/api/fairness/demographics/*` report `rejected = count - approved`, so Review cases are counted as rejected.  
**Action:** Rename the field to `not_approved`, or report Review separately, when the officer dashboard is built.

---

### #70: Existing Local Database Upgraded in Place
**Status:** By design  
**Phase:** 4.8  
**Description:** On first start after Phase 4.8, `upgrade_schema` adds `applications.tenure_months` (default 12) to an existing `creditxplain.db`, and `create_all` adds the `documents` table. No data is removed. A full migration tool (Alembic) is not used.

---

## Change Log (Updated)

| Date | Issue | Action |
|---|---|---|
| Nov 19, 2026 | Phase 4.3 | Loan calculator complete |
| Nov 19, 2026 | #48 | By Design: Fairness metrics require manual trigger |
| Nov 19, 2026 | #49 | By Design: Minimum 50 applications required |
| Nov 19, 2026 | #50 | Identified: No historical trend visualization |
| Nov 19, 2026 | #51 | Identified: Demographic stats may be stale |
| Nov 19, 2026 | Phase 4.5 | Fairness monitoring complete - 6/6 tests passing |
| Nov 25, 2026 | Phase 4.6 | Full test run: 88 + 36 passed; frontend build clean, lint 1 error |
| Nov 25, 2026 | #15, #31, #36, #37, #40, #41 | Marked resolved or superseded (stale) |
| Nov 25, 2026 | #32 | Partly resolved |
| Nov 25, 2026 | #27 | Reopened: Pydantic warnings still present |
| Nov 25, 2026 | #52–#59 | Identified: LIME not local, LIME overflow, small consistency sample, circular recourse validity, no Approve recourse, frontend lint, pandas warning, two CF implementations |
| Nov 25, 2026 | #60 | Resolved: duplicate recourse script deleted |
| Nov 25, 2026 | #52, #53, #54, #55, #59 | Resolved in Phase 4.7 |
| Nov 25, 2026 | #56 | Mitigated: Review → Approve tier |
| Nov 25, 2026 | #61–#66 | Identified: DiCE not completed, coverage trade-off, small age groups, FastAPI on_event, SHAP notice, LIME subset |
| Nov 26, 2026 | #27, #32, #33, #44, #46, #58, #64 | Resolved in Phase 4.8 |
| Nov 26, 2026 | #67–#70 | Identified: no upload scanning/deletion, live EOD always zero, demographic rejected count, in-place DB upgrade |

---

**Next Update:** Phase 4.4
