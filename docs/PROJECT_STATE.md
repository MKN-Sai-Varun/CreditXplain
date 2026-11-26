# PROJECT_STATE — CreditXplain

**Snapshot:** Nov 25, 2026, end of Phase 4.7. Written from the repository (branch `dev-varun`, Phase 4.6 at `dedac96` plus the Phase 4.7 changes) and from test and evaluation runs on the Mac.

**Status legend**

| Mark | Meaning |
|---|---|
| ✅ | Done, with real output or passing tests on the user's machine |
| 🟡 | Works, but has a known gap that matters for the paper or the app |
| ⬜ | Not started |

---

## 1. How to resume

1. Read this file, then `KNOWN_ISSUES.md` #61–#66 (the newest findings).
2. Check the repo: `git log --oneline -5`, `git status --short`.
3. Follow the working agreement in section 6: propose the next phase and wait for approval.

---

## 2. Repository and workspace

| Item | State |
|---|---|
| Remote | `https://github.com/MKN-Sai-Varun/CreditXplain` |
| Working branch | `dev-varun`. Phase 4.6 at `dedac96`; Phase 4.7 committed on top. The user merges into `main` at the end |
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
| Gender and age excluded | ✅ | `CODE_GENDER_M`, `CODE_GENDER_XNA` and `DAYS_BIRTH` excluded; 300 features. Gender ablation: with gender 0.7685, without 0.7668 (cost 0.0017 AUC; `reports/gender_ablation.csv`) |
| Three-band router (`src/decision.py`) | ✅ | Approve below 0.0349, reject from 0.1189. Test bands: approve 34.8% (actual default 2.1%), review 45.0% (6.7%), reject 20.1% (21.6%) (`reports/final_model_bands.csv`) |
| Shared inference (`src/inference.py`) | ✅ | Same feature path for training, evaluation and API (D-39) |
| EDA report | ⬜ | KNOWN_ISSUES #1 |

### Explainability, recourse, fairness (original Phase 2–3)

| Component | Status | Evidence / gap |
|---|---|---|
| SHAP (`src/explainability.py`) | ✅ | Additivity difference 0.0; repeat-run correlation 1.0. Top features: `CREDIT_GOODS_EXT_RISK`, `EXT_SOURCE_MEAN`, `CREDIT_ANNUITY_RATIO` |
| LIME (`src/lime_local.py`) | ✅ | Local, sign-correct, seeded (D-43). On 150 applicants: Spearman with SHAP 0.64 (was 0.00), locality gap 0.49 (was 0.00), sign agreement 0.74, seed stability 0.83, 38 ms (`reports/explanation_consistency.json`). Covers 25 top-gain features (#66) |
| Recourse (`src/recourse.py`) | ✅ | Bounded search over loan amount, income and tenure (D-38) with a safety margin, a plausibility check, refinement, two tiers and SHAP drivers (D-44 to D-48). Usable option: 59.0% of rejected applicants (previous engine 21.3%), 27.3% of Review applicants get a path to Approve; 24% smaller changes than before, 30% smaller than random search; about 20 ms (`reports/recourse_evaluation.json`). DiCE comparison not completed (#61) |
| Counterfactuals, Phase 3 version (`src/counterfactuals.py`, DiCE) | 🟡 | Offline Phase 3 script; DiCE now also runs as a baseline inside `src/recourse_eval.py` |
| Fairness audit (`src/fairness_audit.py`) | ✅ / 🟡 | Age-band DPD 0.349 and EOD 0.350; has-children DPD 0.045. The age gap tracks real default rates (11.4% for under-25 vs 5.2% for 55+) even with age excluded, so it comes through proxies. Mitigation or a documented decision is still open (#16) |

### Application (original Phase 4)

| Component | Status | Evidence / gap |
|---|---|---|
| FastAPI backend, SQLite, JWT login with roles | ✅ | Registration always creates applicants (D-40) |
| Applications, predictions (SHAP + LIME with rules + recourse for Reject and Review, with drivers), loan terms (EMI) | ✅ | `backend/tests/test_predict_api.py`, `test_recourse_api.py`, `test_loan_calculator.py` |
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

## 4. Test baseline (Nov 25, 2026, after Phase 4.7)

| Suite | Result |
|---|---|
| `python3 -m pytest tests backend/tests -q` | ✅ 142 passed (124 before 4.7 + 18 new) |
| `npx tsc --noEmit` (frontend) | ✅ |
| `npm run build` (frontend) | ✅ |
| `npm run lint` (frontend) | ❌ 1 error (#57) |

Warnings to clean up: Pydantic class-based `Config` (#27), pandas `observed=` (#58), FastAPI `on_event` (#64). The matmul RuntimeWarnings are a numpy/macOS artefact (#53).

---

## 5. What stands between the project and a publishable result

The model and the system are in place. Reviewers will look hardest at the claims around the novelty, so these come first:

1. ~~LIME must explain individual applicants~~ ✅ done in 4.7.
2. ~~Recourse needs standard evaluation~~ ✅ done in 4.7, except the DiCE baseline (#61), which needs a 30–40 minute run.
3. **Recourse framing:** a graduated path (Reject → Review → Approve). State the coverage trade-off of the stricter rules (#62) and confirm the age-band gaps on a larger sample (#63).
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

**Next phase (awaiting approval): 4.8, backend gaps:** document upload, officer queue formatting, tenure as an input (D-41). Optionally first: finish the DiCE baseline run (#61).

| # | Open question | Status |
|---|---|---|
| Q1 | Gender excluded from the model? | **Settled by implementation:** excluded (age too); kept for the audit only |
| Q2 | How is the final model reported? | **Settled by implementation:** a single fit, test AUC 0.7668 |
| Q3 | Review and submission dates | Open |
| Q4 | What have Members 2, 3 and 4 built? | Open |
| Q5 | Family status: keep as a feature and audit, or exclude? | Open (age is already excluded) |
| Q7 | Registration: real sign-up or seeded demo applicants only? | Open (real sign-up exists for applicants) |
| Q8 | Recourse in the paper: Reject→Review only, or add a Review→Approve tier? | **Settled:** both tiers (user: 'do the best you can'), implemented in 4.7 |
