# PROJECT_STATE — CreditXplain

**Snapshot:** Nov 25, 2026, end of Phase 4.6. Written from the repository itself (branch `dev-varun` at `b68c3f7` plus the Phase 4.6 working-tree changes) and from a full test run on the Mac. The previous version (Nov 19) was written from conversation memory and was out of date.

**Status legend**

| Mark | Meaning |
|---|---|
| ✅ | Done, with real output or passing tests on the user's machine |
| 🟡 | Works, but has a known gap that matters for the paper or the app |
| ⬜ | Not started |

---

## 1. How to resume

1. Read this file, then `KNOWN_ISSUES.md` #52–#60 (the newest findings).
2. Check the repo: `git log --oneline -5`, `git status --short`.
3. Follow the working agreement in section 6: propose the next phase and wait for approval.

---

## 2. Repository and workspace

| Item | State |
|---|---|
| Remote | `https://github.com/MKN-Sai-Varun/CreditXplain` |
| Working branch | `dev-varun`, at `b68c3f7` before Phase 4.6. The user merges into `main` at the end |
| Machine | MacBook Air, zsh, Python 3.9.6 (`/usr/bin/python3`), pytest 8.4.2, Node 26.7.0 |
| Untracked | `frontend/`: a Next.js 16 / React 19 / Tailwind 4 scaffold plus `lib/api.ts`. Never committed |
| Local-only (gitignored) | `data/`, `creditxplain.db`, `home-credit-default-risk.zip`. Model files in `models/` |

---

## 3. Status by component

### Model (original Phase 1–2)

| Component | Status | Evidence |
|---|---|---|
| Data prep (`src/data_prep.py`) | ✅ | Shape (307511, 133), target rate 0.0807 |
| Baselines LR/RF/XGB/LGBM | ✅ | `reports/baseline_model_comparison.csv` |
| Feature engineering v2 (`src/feature_engineering_v2.py`) | ✅ | Importable; `tests/test_fe_v2.py` |
| Final model (`src/final_model.py`) | ✅ | Single LightGBM fit on 196,806 rows, early stopping on 49,202 validation rows (best iteration 624). Validation AUC 0.7711; **test AUC 0.7668**, PR-AUC 0.2553 (`models/final_model_card.json`) |
| Gender and age excluded | ✅ | `CODE_GENDER_M`, `CODE_GENDER_XNA` and `DAYS_BIRTH` excluded; 300 features. Gender ablation: with gender 0.7685, without 0.7668 (cost 0.0017 AUC; `reports/gender_ablation.csv`) |
| Three-band router (`src/decision.py`) | ✅ | Approve below 0.0349, reject from 0.1189. Test bands: approve 34.8% (actual default 2.1%), review 45.0% (6.7%), reject 20.1% (21.6%) (`reports/final_model_bands.csv`) |
| Shared inference (`src/inference.py`) | ✅ | Same feature path for training, evaluation and API (D-39) |
| EDA report | ⬜ | KNOWN_ISSUES #1 |

### Explainability, recourse, fairness (original Phase 2–3)

| Component | Status | Evidence / gap |
|---|---|---|
| SHAP (`src/explainability.py`) | ✅ | Additivity difference 0.0; repeat-run correlation 1.0. Top features: `CREDIT_GOODS_EXT_RISK`, `EXT_SOURCE_MEAN`, `CREDIT_ANNUITY_RATIO` |
| LIME | 🟡 | **Not local as configured** (#52), numerical overflow (#53), agreement measured on 6 applicants only (58.3% top-10 overlap) (#54) |
| Recourse, API version (`src/recourse.py`) | 🟡 | Bounded search over loan amount, income and tenure (D-38). On 300 rejected test applicants: recourse found for 87.7%, 0% reach Approve, 13.6 ms mean latency. Reported validity of 1.0 is circular (#55, #56) |
| Counterfactuals, Phase 3 version (`src/counterfactuals.py`, DiCE) | 🟡 | Offline only; candidate baseline for the recourse comparison (#59) |
| Fairness audit (`src/fairness_audit.py`) | ✅ / 🟡 | Age-band DPD 0.349 and EOD 0.350; has-children DPD 0.045. The age gap tracks real default rates (11.4% for under-25 vs 5.2% for 55+) even with age excluded, so it comes through proxies. Mitigation or a documented decision is still open (#16) |

### Application (original Phase 4)

| Component | Status | Evidence / gap |
|---|---|---|
| FastAPI backend, SQLite, JWT login with roles | ✅ | Registration always creates applicants (D-40) |
| Applications, predictions (SHAP + LIME + recourse), loan terms (EMI) | ✅ | `backend/tests/test_predict_api.py`, `test_recourse_api.py`, `test_loan_calculator.py` |
| Officer queue, actions, analytics, audit log | 🟡 | Queue formatting TODO at `backend/routers/officer.py:21`; no API tests for officer routes (#32) |
| Fairness monitoring API | ✅ | 4 officer-only endpoints, manual trigger (#48) |
| Document upload | ⬜ | Officers can request documents, but applicants cannot upload them |
| PDF reports | ⬜ | Not implemented (ReportLab planned) |
| Loan tenure input | ⬜ | Fixed at 12 months (#44) |
| Frontend: applicant portal, officer dashboard | ⬜ | Starter page only; lint fails with 1 error (#57) |

### Testing and documentation (original Phase 5)

| Item | Status |
|---|---|
| CI | ⬜ No `.github/` workflow |
| README | ⬜ Out of date |
| End-to-end latency, usability check | ⬜ |
| Manuscript | ⬜ |

---

## 4. Test baseline (Nov 25, 2026)

| Suite | Result |
|---|---|
| `python3 -m pytest tests -q` | ✅ 88 passed |
| `python3 -m pytest backend/tests -q` | ✅ 36 passed |
| `npx tsc --noEmit` (frontend) | ✅ |
| `npm run build` (frontend) | ✅ |
| `npm run lint` (frontend) | ❌ 1 error (#57) |

Warnings to clean up: Pydantic class-based `Config` (#27, reopened), pandas `observed=` (#58), LIME RuntimeWarnings (#53).

---

## 5. What stands between the project and a publishable result

The model and the system are in place. Reviewers will look hardest at the claims around the novelty, so these come first:

1. **LIME must explain individual applicants** before any SHAP–LIME comparison is reported (#52–#54).
2. **Recourse needs standard evaluation:** coverage, proximity, sparsity, plausibility, latency, and a baseline under the same bounds. The circular validity rate must go (#55).
3. **Recourse framing:** present it as Reject→Review, and state that Approve is out of reach because the strongest risk drivers (bureau scores) are not actionable (#56).
4. **Fairness:** report the age disparity with per-group default rates and AUC, and decide on mitigation or a documented justification (#16).
5. **A working demo:** frontend, PDF export, and the remaining backend gaps.

Phases 4.7 onward (TECHNICAL_DECISIONS D-41) follow this order.

---

## 6. Working agreement (user, Nov 25, 2026)

- Work **phase by phase**. Propose each phase and **do nothing without approval**.
- After a phase: update the four docs, **explain every change**, and give the tests to run.
- **The user commits and pushes** to `dev-varun`. The assistant suggests commit commands that add files by name (never `git add .`) and does not run them.
- Keep code comments minimal and natural. Verify before asserting.
- Do not commit data, models, `.env`, the SQLite database or per-applicant prediction dumps.

---

## 7. Next phase and open questions

**Next phase (awaiting approval): 4.7, publication-grade evaluation of LIME and recourse**, following the proposed standard in D-42.

| # | Open question | Status |
|---|---|---|
| Q1 | Gender excluded from the model? | **Settled by implementation:** excluded (age too); kept for the audit only |
| Q2 | How is the final model reported? | **Settled by implementation:** a single fit, test AUC 0.7668 |
| Q3 | Review and submission dates | Open |
| Q4 | What have Members 2, 3 and 4 built? | Open |
| Q5 | Family status: keep as a feature and audit, or exclude? | Open (age is already excluded) |
| Q7 | Registration: real sign-up or seeded demo applicants only? | Open (real sign-up exists for applicants) |
| Q8 | Recourse in the paper: accept Reject→Review as the claim, or add a Review→Approve tier? | New, for 4.7 |
