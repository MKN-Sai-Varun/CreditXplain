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

### #2: SHAP/LIME Not Implemented
**Status:** Open  
**Phase:** 2  
**Description:** Phase 2 requires SHAP (global summary, per-applicant waterfall) and LIME (secondary comparison), but `src/explainability.py` does not exist.  
**Impact:** Objective O5 incomplete (model selection requires accuracy + explainability-consistency). Cannot proceed to Phase 3 counterfactual engine without baseline explanations.  
**Action:** Implement in next phase after model lock.  
**Owner:** Member 2 (per team roles)

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

### #4: Uncommitted Experimental Files
**Status:** Open  
**Phase:** 2  
**Description:** Several files are untracked or have unstaged deletions:
- `src/evaluate_on_test.py` (untracked, should be committed)
- `src/train_lightgbm_recall.py` (untracked, obsolete per PROJECT_STATE.md)
- `src/lgbm_capacity_experiment.py` (untracked, broken: never loads target)
- Stale reports: `lightgbm_cost_optimized.csv`, `lightgbm_cost_threshold_sweep.csv`, `lightgbm_recall_optimized.csv`
- `backend/models/model.py` (deleted in working tree, deletion not staged)

**Impact:** Repo hygiene; confusion over which files are current.  
**Action:** Stage `evaluate_on_test.py`; delete obsolete files; stage `backend/models/model.py` deletion.  
**Owner:** Varun (pending in Phase 1 cleanup)

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

### #8: Gender Feature Importance in Existing Models
**Status:** Noted  
**Phase:** 2  
**Description:** `lgbm_fe_v2_cv.py` model (before exclusion) included CODE_GENDER_M with gain importance 4219.66 (1.02%). This is now excluded in `final_model.py`.  
**Impact:** Ablation study will quantify performance impact (expected: minimal).  
**Action:** Run `--ablation` flag in Phase 1 to document difference.  
**Owner:** Varun (Phase 1 execution)

---

### #9: Stale Cost-Optimized Reports
**Status:** Open  
**Phase:** 2  
**Description:** `reports/lightgbm_cost_optimized.csv`, `lightgbm_cost_threshold_sweep.csv`, `lightgbm_recall_optimized.csv` are from SMOTE-era experiments and are superseded by the three-band router.  
**Impact:** Confusion over which results are current.  
**Action:** Delete or move to `reports/archive/` in Phase 1 cleanup.  
**Owner:** Varun

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

### #11: lgbm_capacity_experiment.py Broken
**Status:** Open  
**Phase:** 2  
**Description:** Script reads only feature file (`X_train_fe_v2.csv`) but never loads target (`y_train.csv`), so any fit attempt fails.  
**Action:** Delete (obsolete experiment) or fix if needed.  
**Owner:** Varun

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
| Nov 19, 2026 | All | Initial KNOWN_ISSUES.md created |

---

**Next Update:** After Phase 1 tests complete and issues are triaged
