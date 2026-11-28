# MASTER_PLAN — CreditXplain

**Read this first.** This file has four parts, and they must not be mixed up:

| Part | What it is | Authority |
|---|---|---|
| **A** | The **original roadmap**, taken from the team's planning document (`Plan`) | Original. Do not replace it |
| **B** | **Amendments made later** by the project lead | Project-lead decisions |
| **C** | A **working breakdown** (8 phases), proposed later | **Proposal only; never formally ratified** |
| **D** | Status of the original phases | Derived from `PROJECT_STATE.md` |

Companion files: `PROJECT_STATE.md` (where we stopped), `IMPLEMENTATION_LOG.md` (what was done), `TECHNICAL_DECISIONS.md` (stack and design), `KNOWN_ISSUES.md` (bugs, gaps, unrecoverable information).

---

## 0. Project identity

- **Name:** CreditXplain, *An Explainable, Counterfactual-Driven Decision Support System for Credit Risk Prediction*
- **Context:** B.Tech final-year major project, KMIT Hyderabad, academic year 2026–2027. Target venue: IEEE Access (or a comparable Q1/Q2 venue). 4-member team, 24 weeks.
- **Repository:** `https://github.com/MKN-Sai-Varun/CreditXplain`. License: MIT.
- **Base paper extended:** Shreya & Pathak (2025), arXiv:2506.19383. The citation was checked against the arXiv page. The paper reports: Random Forest accuracy 82.03% / ROC-AUC 0.6808; XGBoost 88.74% / 0.7033; LightGBM 90.07% / 0.7203 (top performer).
- **Dataset:** Home Credit Default Risk (Kaggle). Secondary option in the plan: Give Me Some Credit.

### Capabilities the project adds over the base paper (the novelty)

| Capability | Base paper | CreditXplain |
|---|---|---|
| Risk prediction + SHAP/LIME | Yes | Retained as baseline |
| Actionable recourse for rejected applicants | No | **Counterfactual Recommendation Engine (primary novelty)** |
| Fairness / bias audit across subgroups | No | **Explicit subgroup audit (Fairlearn / AIF360)** |
| Interactive live decision-support tool | No (static HTML/JSON/PNG) | **Interactive app + PDF export** |

Closest competing work, per the plan: Kapoor et al. (2026), FICO dataset, TabNet + LightGBM + SHAP + counterfactuals. The plan says to read it in full before submission.

---

# PART A — ORIGINAL ROADMAP (from the planning document)

## A.1 Objectives

| ID | Objective |
|---|---|
| **O1** | Reproduce the base paper's preprocessing, SMOTE-based imbalance handling, and RF/XGBoost/LightGBM + SHAP/LIME pipeline as the project baseline |
| **O2** | Design and implement a Counterfactual Recommendation Engine for applicant recourse |
| **O3** | Build an interactive banking dashboard (risk scores, explanations, historical analytics, PDF export) |
| **O4** | Conduct a fairness/bias audit across applicant subgroups and document disparities honestly |
| **O5** | Compare Logistic Regression, Random Forest, XGBoost and LightGBM; select the best model using accuracy **and** explainability-consistency, not accuracy alone |
| **O6** | Produce a reproducible codebase and a manuscript suitable for peer-reviewed submission |

## A.2 The five phases, in original order

The plan states acceptance as "Key Deliverable" per phase and gives no further numeric criteria. Where a criterion below is not in the plan, it is marked *(not specified in plan)*.

### Phase 1 — Literature Review & Dataset Prep (Weeks 1–4)
- **Objectives:** O1 (preprocessing part). Deep review of the base paper plus 8–10 related works; acquire and clean the dataset; EDA.
- **Tasks (from plan §4 and §5.2):**
  1. Acquire Home Credit data via the Kaggle CLI.
  2. Load only the scoped tables: `application_train.csv`, `bureau.csv`, `previous_application.csv`. The other four tables are stretch goals.
  3. Aggregate `bureau` and `previous_application` to one row per `SK_ID_CURR` before merging (count, mean credit, max overdue, refused count).
  4. Clean: flag then null the `DAYS_EMPLOYED == 365243` placeholder; median/mode imputation; 3σ outlier clipping.
  5. Engineer `CREDIT_TO_GOODS_RATIO`, `AGE_YEARS`, `CREDIT_INCOME_RATIO`.
  6. Simulate application timestamps (for the dashboard's historical analytics; must be documented as simulated).
  7. EDA (class-imbalance check; age histogram saved to `reports/eda_age_distribution.png`).
  8. Literature review and gap table (plan §8–9).
- **Dependencies:** Kaggle account and competition rules accepted.
- **Key deliverable (acceptance):** finalized gap table, cleaned dataset, EDA report.

### Phase 2 — Baseline Models & Explainability (Weeks 5–9)
- **Objectives:** O1, O5.
- **Tasks (plan §5.3):**
  1. 80/20 stratified split; SMOTE on training data (per base paper).
  2. Train and compare LR, RF, XGBoost, LightGBM; report accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
  3. Select the best model using accuracy + explainability-consistency.
  4. SHAP (primary): `TreeExplainer`, global summary plot, per-applicant waterfall.
  5. LIME (secondary comparison) on the same applicants.
- **Dependencies:** Phase 1 cleaned dataset.
- **Key deliverable:** best model selected; SHAP/LIME pipeline working.

### Phase 3 — Counterfactual Engine & Fairness (Weeks 10–13)
- **Objectives:** O2, O4.
- **Tasks (plan §3.3, §5.4):**
  1. Counterfactual Recommendation Engine: bounded-perturbation search over actionable features to find the smallest realistic change-set that flips the decision. The plan's scaffold uses DiCE (`method="random"`) with actionable features `AMT_CREDIT`, `CREDIT_INCOME_RATIO`, `AMT_INCOME_TOTAL`, `CNT_FAM_MEMBERS`, `DAYS_EMPLOYED`. The architecture text also names income, existing debt, tenure, co-applicant status. Include a domain-informed bounds step and a feasibility-ranking step (plan §7).
  2. Fairness audit: Demographic Parity Difference and Equal Opportunity Difference by subgroup (Fairlearn or AIF360), restricted to attributes appropriate for the public dataset.
- **Dependencies:** Phase 2 selected model.
- **Key deliverable:** working recommendation engine; fairness audit report.

### Phase 4 — Dashboard Development (Weeks 14–18)
- **Objectives:** O3.
- **Tasks (plan §3, §5.5):** three-lane application, matching the workflow diagram.
  - *Applicant portal:* login/register, submit application, live status, view decision + explanation + recourse, upload requested documents.
  - *Backend:* ingestion and validation → risk prediction engine → explainability layer → **decision router (Approve / Review / Reject)**; Approve → loan terms and interest-rate calculation; Reject → counterfactual engine; Review → officer queue; PDF report generator.
  - *Fairness & bias monitor* (parallel branch): subgroup metrics → compliance alert to the officer dashboard. Does not block the decision path.
  - *Officer dashboard:* manual review queue; review risk score + explanation; officer action (confirm / override with logged justification / request documents); historical prediction analytics; compliance alert panel; officer-triggered PDF to the compliance archive.
  - SQLite persistence shared by portal and dashboard.
- **Key deliverable:** integrated decision-support application. *(The original plan names Streamlit; see B1.)*

### Phase 5 — Testing & Documentation (Weeks 19–24)
- **Objectives:** O6.
- **Tasks:** system testing against the Section 6 metrics; manuscript drafting with real results; code release; submission.
- **Key deliverable:** evaluation results, manuscript, code release, submission.

## A.3 Team roles (plan §2.3)

| Member | Responsibility |
|---|---|
| Member 1 (Varun) | Data preprocessing, EDA, baseline model implementation and comparison |
| Member 2 | SHAP/LIME integration, global/local visualization |
| Member 3 | Counterfactual Recommendation Engine (core novelty) |
| Member 4 | Dashboard, fairness analysis, PDF reporting, manuscript coordination |

All members are to be cross-trained on the evaluation pipeline. Pranav is the teammate seen in the git history (`dev-pranav`); which numbered member he is was not stated.

## A.4 Evaluation methodology (plan §6)

1. **Predictive performance:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix, selected model vs all baselines.
2. **Explainability and recommendation quality:** SHAP consistency across repeated runs and vs LIME; **counterfactual validity rate** (share of recommendations that are realistic, actionable and actually flip or improve the prediction); **prediction latency** (input submission to rendered explanation).
3. **Fairness:** Demographic Parity Difference and Equal Opportunity Difference across permissible subgroups.
4. **Human interpretability:** small usability check (team plus an independent reviewer where feasible) on whether SHAP views and recommendations are understandable to a non-technical loan officer.
5. **Baselines to compare:** LR (added as a reference), RF, XGBoost, LightGBM, with the base paper's reported numbers above.

## A.5 Risks listed in the plan (§7)

Class imbalance (High); overfitting of boosted models; SHAP cost at scale; unrealistic counterfactuals (domain bounds plus feasibility ranking); sensitive-attribute handling in the fairness audit (restrict to appropriate attributes; consult guidance); dashboard scope creep (freeze the feature list at the end of Phase 3); results below hypothesised targets (report honestly).

## A.6 Scope control (plan §4.2–4.3)

Replicate only the merge/aggregation needed to reproduce the base paper's top features (`EXT_SOURCE_1/2/3`, `CREDIT_TO_GOODS_RATIO`, `CNT_FAM_MEMBERS`, `AGE_YEARS`, `YEARS_BEGINEXPLUATATION_MEDI`). `bureau_balance`, `POS_CASH_balance`, `credit_card_balance` and `installments_payments` are stretch goals.

**The plan's "Implementation TimeLine" heading at the end of the document is empty.** No dates were ever recorded in it.

---

# PART B — AMENDMENTS MADE AFTER THE ORIGINAL PLAN

| ID | Amendment | Source |
|---|---|---|
| **B1** | Frontend changed from Streamlit to **React / Next.js**, with a **FastAPI** backend and **SQLite**. Streamlit is fully retired | Project-lead decision |
| **B2** | Dev work happens on branch **`dev-varun`**; merged into `main` at the end | Project-lead decision |
| **B3** | Design the application as if for a **single user**, following the workflow diagram | Project-lead decision |
| **B4** | Work **phase by phase**, with tests after each phase, and commit steps | Project-lead decision |
| **B5** | **Gender must not be taken into consideration by the model or its explanations**, because it is not a factor in deciding credit | Project-lead decision. Implemented as: gender excluded from the model entirely and kept only for the fairness audit (D-3; confirmed, KNOWN_ISSUES Q1) |
| **B6** | **Leaning toward a recall-optimised threshold**. Not formally settled; see Part C D-2 | Project lead |
| **B7** | Progress and decision tracking lives in **Notion**. The repo keeps only code, the README and result summaries | Project-lead decision |
| **B8** | Design choices not stated above are delegated | Project-lead decision |

---

# PART C — WORKING BREAKDOWN PROPOSED LATER (NOT THE ORIGINAL ROADMAP)

This breakdown was proposed late in planning. The follow-up direction "go phase wise 4.1, 4.2 ..." is **ambiguous**: it may mean sub-step numbering inside a phase, or a different numbering scheme. The breakdown has not been confirmed. If the original five phases are kept as the only structure, use Part A and treat this table as sub-tasks.

| # | Proposed phase | Maps to original | Test gate (proposed) |
|---|---|---|---|
| 1 | **Lock the model:** held-out test scores for FE v2; one final model saved with its thresholds and a model card; FE v2 made importable and tested | Phase 2 (selection) | `test_fe_v2.py`, `test_final_model.py`, `test_decision.py` green; saved model reproduces its reported test AUC |
| 2 | **One reproducible pipeline:** a single run order, README rewritten, ablation table, CI running pytest | Phase 2 / 5 | A fresh clone runs end to end; the two open PR #3 review findings closed |
| 3 | **Explainability:** SHAP global and per-applicant; LIME on the same applicants | Phase 2 | SHAP additivity test; stable repeat runs |
| 4 | **Router, loan terms, counterfactual engine** | Phase 3 (+ diagram) | Recourse flips the decision, respects bounds, never changes immutable features; EMI matches known values; rate never decreases as risk rises |
| 5 | **Fairness audit** | Phase 3 | Metrics match hand-computed toy values |
| 6 | **Backend API** (FastAPI, SQLite, login) | Phase 4 | API tests green; latency measured |
| 7 | **Frontend** (Next.js applicant portal and officer dashboard) | Phase 4 | Build and lint clean; manual checklist |
| 8 | **Evaluation and manuscript** | Phase 5 | Every number in the paper traces to a report file |

Proposed decisions (delegated by the project lead; none was explicitly confirmed):

| ID | Decision | Basis |
|---|---|---|
| D-1 | Final model = **single LightGBM** on FE v2 features, not the LightGBM+CatBoost blend | Blend gave +0.0010 AUC in CV and was lower on test (0.7720 vs 0.7722). **Caveat in KNOWN_ISSUES #3:** those test scores are fold-averages, not single models |
| D-2 | **Three-band router** instead of one threshold: Approve below a low cutoff, Reject from a higher cutoff, Review in between | Matches the diagram's Decision Router and settles the 0.08-vs-0.162 argument. Cutoff values are **not yet computed on the real data** |
| D-3 | **Gender excluded from the model**, saved separately for the fairness audit only. Age and family status stay as features for now and are audited by band | Follows B5 |
| D-4 | Evaluation protocol: train (fit) / validation (early stopping and cutoffs) / test (reported once) | Existing thresholds in `train_baseline.py` were chosen on the test set |

Additional component the diagram requires that the original text under-specifies: **Loan Terms & Interest Rate Calculation** (Approve branch). Proposed as a transparent rule layer, because Home Credit has no interest rates to learn from: a risk-band to rate-spread table plus a standard EMI formula. The EMI test value proposed was ₹1,00,000 at 12% a year over 12 months = 8,884.88. The rates are illustrative assumptions and must be labelled that way.

---

# PART D — STATUS OF THE ORIGINAL PHASES

Full detail is in `PROJECT_STATE.md`. Legend: ✅ complete and verified by real output · 🟡 partly done · ⬜ not started.

| Original phase | Status |
|---|---|
| 1. Literature & dataset prep | ✅ cleaned dataset verified on real data; EDA report in `reports/eda/` (Phase 5.6). The gap/novelty tables are in the plan itself |
| 2. Baseline models & explainability | ✅ Final LightGBM (test AUC 0.7668, gender excluded; age kept as a feature and audited, D-3). SHAP verified; local LIME with SHAP agreement measured on 150 applicants (Phase 4.7) |
| 3. Counterfactual engine & fairness | ✅ Two-tier recourse with safety margin and plausibility check, evaluated against the previous engine, random search, DiCE and ablations. Offline fairness audit done; age disparity documented |
| 4. Dashboard | ✅ FastAPI backend complete (auth, applications, predictions, recourse, loan terms, documents, officer workflow, fairness monitoring; Phase 4.8) and PDF reports (Phase 4.4); Next.js applicant portal (Phase 4.9) and officer dashboard (Phase 4.10). Phase 4 complete |
| 5. Testing & documentation | 🟡 Automated tests, CI and a rewritten README with a reproducible pipeline (Phase 5.1). Recourse robustness (5.2–5.2c), latency (5.3) and understandability (5.4: computed measures plus a prepared expert review) age as eligibility rules with the age experiment (5.5a) the EDA report (5.6), leak-free cleaning (5.6b) the paper's figures and tables (5.7) and the full manuscript draft (5.8) done. Author details, reference checks, release and submission remain |
