# KNOWN_ISSUES — CreditXplain

**Last Updated:** Nov 19, 2026 (Phase 1 implementation)

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
**Action:** User decision required (deferred to Phase 5).  
**User Input Needed:** Yes

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

### #5: Copilot Medium Findings (Open)
**Status:** Open  
**Phase:** 2  
**Source:** GitHub Copilot review on PR #3

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

## ❓ Open Questions (User Input Needed)

### Q1: ~~Gender Exclusion Confirmation~~ ✅ **RESOLVED**
**Resolved:** Nov 19, 2026  
**User Guidance:** "Gender should not be taken into consideration by the model"  
**Decision:** Exclude CODE_GENDER entirely from model (D-3 in TECHNICAL_DECISIONS.md)

---

### Q2: Age and Family Status Handling
**Status:** Open  
**Phase:** 3  
**Question:** Should age and family status also be excluded from the model (like gender), or kept and audited?  
**User Guidance (Nov 19):** "Age is a consideration, since if they are minors, guardians are required. For minors or students, father's employment and securities are necessary."  
**Decision:** Keep AGE_YEARS and family status features; audit for fairness disparities in Phase 3.  
**Action:** Monitor for age-based disparities beyond legal thresholds.

---

### Q3: Review Timeline and Submission Dates
**Status:** Open  
**Phase:** 5  
**Question:** What are the real project review dates and submission deadline?  
**Description:** MASTER_PLAN timeline section is empty; no dates recorded.  
**Impact:** Cannot plan Phase 5 manuscript completion.  
**Action:** User to provide academic calendar milestones.

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
| Nov 19, 2026 | Q1 | Resolved: Gender exclusion confirmed by user |
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
