# PROJECT_STATE — CreditXplain

**Snapshot:** Nov 26, 2026, end of Phase 4.8. Written from the repository (branch `dev-varun` at `f1b8bc2` plus the Phase 4.8 changes) and from the full test run.

**Status legend**

| Mark | Meaning |
|---|---|
| ✅ | Done, with real output or passing tests on the development machine |
| 🟡 | Works, but has a known gap that matters for the paper or the app |
| ⬜ | Not started |

---

## 1. How to resume

1. Read this file, then `KNOWN_ISSUES.md` #61–#75 (the newest findings).
2. Check the repo: `git log --oneline -5`, `git status --short`.
3. Follow the working agreement in section 6: propose the next phase and wait for approval.

---

## 2. Repository and workspace

| Item | State |
|---|---|
| Remote | `https://github.com/MKN-Sai-Varun/CreditXplain` |
| Working branch | `dev-varun`: Phase 4.6 `dedac96`, Phase 4.7 `473917f`, docs wording `f1b8bc2`. Merged into `main` at the end |
| Machine | MacBook Air, zsh, Python 3.9.6 (`/usr/bin/python3`), pytest 8.4.2, Node 26.7.0 |
| Untracked | `frontend/`: a Next.js 16 / React 19 / Tailwind 4 scaffold plus `lib/api.ts`. Never committed |
| Local-only (gitignored) | `data/`, `creditxplain.db`, `home-credit-default-risk.zip`, `models/*.pkl`, `models/recourse_reference.npz` (rebuild with `python3 -m src.build_recourse_reference`) |

---

## 3. Status by component

### Model (original Phase 1–2)

| Component | Status | Evidence |
|---|---|---|
| Data prep (`src/data_prep.py`) | ✅ | Shape (307511, 133), target rate 0.0807 |
| Baselines LR/RF/XGB/LGBM | ✅ | `reports/baseline_model_comparison.csv` |
| Feature engineering v2 (`src/feature_engineering_v2.py`) | ✅ | Importable; `tests/test_fe_v2.py` |
| Final model (`src/final_model.py`) | ✅ | Single LightGBM fit on 196,806 rows, early stopping on 49,202 validation rows (best iteration 624). Validation AUC 0.7711; **test AUC 0.7668**, PR-AUC 0.2553 (`models/final_model_card.json`) |
| Gender excluded; age kept and audited | ✅ | `CODE_GENDER_M`, `CODE_GENDER_XNA` and the raw `DAYS_BIRTH` excluded; age stays in as `AGE_YEARS`, `AGE_SQUARED` and `AGE_CUBED` (D-3: keep age and audit it); 300 features. Gender ablation: with gender 0.7685, without 0.7668 (cost 0.0017 AUC; `reports/gender_ablation.csv`) |
| Three-band router (`src/decision.py`) | ✅ | Approve below 0.0349, reject from 0.1189. Test bands: approve 34.8% (actual default 2.1%), review 45.0% (6.7%), reject 20.1% (21.6%) (`reports/final_model_bands.csv`) |
| Shared inference (`src/inference.py`) | ✅ | Same feature path for training, evaluation and API (D-39) |
| EDA report | ⬜ | KNOWN_ISSUES #1 |

### Explainability, recourse, fairness (original Phase 2–3)

| Component | Status | Evidence / gap |
|---|---|---|
| SHAP (`src/explainability.py`) | ✅ | Additivity difference 0.0; repeat-run correlation 1.0. Top features: `CREDIT_GOODS_EXT_RISK`, `EXT_SOURCE_MEAN`, `CREDIT_ANNUITY_RATIO` |
| LIME (`src/lime_local.py`) | ✅ | Local, sign-correct, seeded (D-43). On 150 applicants: Spearman with SHAP 0.64 (was 0.00), locality gap 0.49 (was 0.00), sign agreement 0.74, seed stability 0.83, 38 ms (`reports/explanation_consistency.json`). Covers 25 top-gain features (#66) |
| Recourse (`src/recourse.py`) | ✅ | Bounded search over loan amount, income, tenure and waiting (3–12 months, age held fixed; D-38, D-56) with a safety margin, a plausibility check, refinement, two tiers, SHAP drivers (D-44 to D-48) and applicant preferences (D-57). Usable option with waiting: 67.0% of rejected applicants and 43.7% of Review applicants (loan levers only: 59.0% and 34.0%; previous engine 21.3%; DiCE 5% on 20 applicants). Changes are 24% smaller than the previous engine's and about 30% smaller than random search's; about 20 ms (`reports/recourse_evaluation.json`, IMPLEMENTATION_LOG 4.7b) |
| Counterfactuals, Phase 3 version (`src/counterfactuals.py`, DiCE) | 🟡 | Offline Phase 3 script; DiCE now also runs as a baseline inside `src/recourse_eval.py` |
| Fairness audit (`src/fairness_audit.py`) | ✅ / 🟡 | Age-band DPD 0.349 and EOD 0.350; has-children DPD 0.045. The age gap tracks real default rates (11.4% for under-25 vs 5.2% for 55+) and the model uses age directly (`AGE_YEARS` and its square and cube); only the raw `DAYS_BIRTH` column is dropped. Mitigation or a documented decision is still open (#16) |

### Application (original Phase 4)

| Component | Status | Evidence / gap |
|---|---|---|
| FastAPI backend, SQLite, JWT login with roles | ✅ | Registration always creates applicants (D-40) |
| Applications, predictions (SHAP + LIME with rules + recourse for Reject and Review, with drivers), loan terms (EMI) | ✅ | `backend/tests/test_predict_api.py`, `test_recourse_api.py`, `test_loan_calculator.py` |
| Officer queue, actions, analytics, audit log | ✅ | Queue returns full details; overrides need a justification; approval creates loan terms (D-51, D-54); `test_officer_api.py` |
| Fairness monitoring API | ✅ | 4 officer-only endpoints, manual trigger (#48) |
| Document upload | ✅ | Upload, list and download with ownership, type, size and signature checks (D-50); `test_documents_api.py`. No scanning or deletion (#67) |
| PDF reports | ✅ | Applicant decision report and officer compliance report (D-59); `test_reports_api.py` |
| Loan tenure input | ✅ | 12 to 60 months chosen by the applicant; loan amount required (D-52, D-53); `test_applications_api.py` |
| Frontend: applicant portal, officer dashboard | ⬜ | Starter page only; lint fails with 1 error (#57) |

### Testing and documentation (original Phase 5)

| Item | Status |
|---|---|
| CI | ⬜ No `.github/` workflow |
| README | ⬜ Out of date |
| End-to-end latency, usability check | ⬜ |
| Manuscript | ⬜ |

---

## 4. Test baseline (Nov 26, 2026, after Phase 4.4)

| Suite | Result |
|---|---|
| `python3 -m pytest tests backend/tests -q` | ✅ 211 passed (201 before 4.4 + 10 new) |
| `npx tsc --noEmit` (frontend) | ✅ |
| `npm run build` (frontend) | ✅ |
| `npm run lint` (frontend) | ❌ 1 error (#57) |

The Pydantic (#27), pandas (#58) and FastAPI (#64) warnings are gone. Remaining warnings come from libraries: matplotlib/pyparsing deprecations, the SHAP output notice (#65), and the numpy/macOS matmul artefact (#53).

---

## 5. What stands between the project and a publishable result

The model and the system are in place. Reviewers will look hardest at the claims around the novelty, so these come first:

1. ~~LIME must explain individual applicants~~ ✅ done in 4.7.
2. ~~Recourse needs standard evaluation~~ ✅ done in 4.7 and 4.7b, including DiCE.
3. **Recourse framing:** a graduated path (Reject → Review → Approve). State the coverage trade-off of the stricter rules (#62) and confirm the age-band gaps on a larger sample (#63).
4. **Fairness:** report the age disparity with per-group default rates and AUC, and decide on mitigation or a documented justification (#16).
5. **A working demo:** backend and PDF reports complete; the frontend remains.

Phases 4.7 onward (TECHNICAL_DECISIONS D-41) follow this order.

---

## 6. Working agreement (Nov 25, 2026)

- Work **phase by phase**. Each phase is approved by the project lead **before any change**.
- After a phase: update the four docs, **document every change**, and list the tests to run.
- **The project lead runs the tests, stages files by name** (never `git add .`), **commits and pushes** to `dev-varun`.
- Keep code comments minimal and natural. Verify before asserting.
- Do not commit data, models, `.env`, the SQLite database or per-applicant prediction dumps.

---

## 7. Next phase and open questions

**Next phase (awaiting approval): 4.9, frontend foundation and applicant portal**, then 4.10 (officer dashboard).

| # | Open question | Status |
|---|---|---|
| Q1 | Gender excluded from the model? | **Settled by implementation:** gender excluded, kept for the audit only. Age is **not** excluded: it stays in as `AGE_YEARS` (and square, cube) and is audited |
| Q2 | How is the final model reported? | **Settled by implementation:** a single fit, test AUC 0.7668 |
| Q3 | Review and submission dates | Open |
| Q4 | What have Members 2, 3 and 4 built? | Open |
| Q5 | Age and family status: keep as features and audit, or exclude? | Open. Both are currently kept and audited (age-band DPD 0.349, #16) |
| Q7 | Registration: real sign-up or seeded demo applicants only? | Open (real sign-up exists for applicants) |
| Q8 | Recourse in the paper: Reject→Review only, or add a Review→Approve tier? | **Settled:** both tiers (project-lead decision), implemented in 4.7 |
