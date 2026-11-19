# PROJECT_STATE — CreditXplain

Snapshot: written at the end of a very long conversation (system date 19 Nov 2026). Everything below is taken from what was pasted or reported in that conversation. Nothing was re-checked against the repository. A fresh conversation should **first read the repo** (it is connected to the Claude Project) and compare it with this file.

**Status legend**

| Mark | Meaning |
|---|---|
| ✅ | Completed, with real output or test results from the user's machine seen in the conversation |
| 🟡 | Implemented and run on real data, but there is no automated test |
| 🧪 | Written and checked **only in the assistant's sandbox on synthetic data**. Never given to the user in final form and **not in the repo** |
| ⬜ | Pending |
| ❓ | Unknown. Cannot be determined from the conversation |

---

## 1. How to resume (read in this order)

1. This file, then `KNOWN_ISSUES.md`.
2. Read the repo: `git log --oneline -8`, `git status --short`, `ls src tests reports models`.
3. Do not start anything before the user answers the open questions in section 7. Question Q1 and Q2 block the next task.

---

## 2. Repository and workspace state

| Item | State |
|---|---|
| Remote | `https://github.com/MKN-Sai-Varun/CreditXplain` |
| `main` | At `3cf328d` (merge of PR #4). Includes merged PRs #2, #3, #4 |
| Working branch | **`dev-varun`**, fast-forwarded to `3cf328d` and pushed. The user is working here and will merge at the end |
| Other remote branches | `origin/dev-pranav` (already merged), `origin/experiments-fe-v2-ensemble` (merged; local copy may still exist) |
| Open PRs | None known |
| Main device | MacBook Air, repo at `/Users/mknsaivarun/Desktop/Varun Files/Projects/CreditXplain`, zsh, system Python 3.9.6 (`/Library/Developer/CommandLineTools`). A Windows machine was used earlier (PowerShell, `C:\Users\mknsa\OneDrive\Desktop\Varun Files\Projects\CreditExplain\Source\creditxplain`) |
| Safety backup | `~/creditxplain_backup/src` and `~/creditxplain_backup/reports` (copied during the cleanup, before the experiment files were committed) |

### Uncommitted local files (as last reported; verify with `git status --short`)

| Path | State | Recommended action (assistant's recommendation; user had not acted yet) |
|---|---|---|
| `backend/models/model.py` | Deleted in the working tree, deletion not staged. Still present on `main` | Stage the deletion in the next commit; superseded |
| `src/train_lightgbm_recall.py` | Untracked | Delete. Superseded by the three-band router. Reads SMOTE-era data, docstring outdated |
| `src/lgbm_capacity_experiment.py` | Untracked | Delete or fix. It reads only the feature file and never loads a target (line 11), so it fails |
| `reports/lightgbm_cost_optimized.csv` | Untracked | Delete. Stale (see KNOWN_ISSUES #9) |
| `reports/lightgbm_cost_threshold_sweep.csv` | Untracked | Delete. Stale |
| `reports/lightgbm_recall_optimized.csv` | Untracked | Delete. Stale |
| `src/evaluate_on_test.py`, `reports/fe_v2_test_results.csv` | ❓ The user ran `python3 src/evaluate_on_test.py` successfully, so the script exists locally. Whether it is committed is unknown. It was not in the `main` file list | Check `git status`; commit if untracked |

Gitignored data and models exist locally (not in git): see `TECHNICAL_DECISIONS.md` section 9.

---

## 3. Status by original phase and task

| Original phase | Task | Status | Evidence |
|---|---|---|---|
| 1 | Acquire Home Credit data | ✅ | 10 CSVs extracted into `data/home_credit/` on the Mac |
| 1 | `src/data_prep.py` | ✅ | Real run: shape `(307511, 133)`, TARGET rate 0.08072881945686496, 0 NaNs, 0 leaked `365243` (matched on Windows and Mac) |
| 1 | EDA report (`reports/eda_age_distribution.png`) | ❓ / ⬜ | No EDA output was ever shown |
| 1 | Literature review / gap tables | ✅ in the plan document (§8–9); no further literature work seen |
| 2 | Baseline comparison LR/RF/XGB/LGBM (`src/train_baseline.py`) | ✅ | `reports/baseline_model_comparison.csv` (numbers in IMPLEMENTATION_LOG §5) |
| 2 | Threshold diagnostics (`src/threshold_check.py`) | 🟡 | Run by the user; confusion matrices reported. Committed version only covers 3 models |
| 2 | Scaling stage (`src/scale_and_balance.py`) | ✅ | Real run: train `(246008, 237)`, rate 0.0807, test `(61503, 237)`. `tests/test_scale_and_balance.py` **3 passed** on the Mac |
| 2 | Feature engineering v2 (`src/feature_engineering_v2.py`) | 🟡 | Real run; CV results exist. No unit test; **not importable** without running the pipeline (module-level code, no main guard) |
| 2 | LightGBM + FE v2, 5-fold CV (`src/lgbm_fe_v2_cv.py`) | 🟡 | `reports/lgbm_fe_v2_results.csv`, OOF AUC 0.771558 |
| 2 | CatBoost + FE v2 (`src/catboost_fe_v2_cv.py`) | 🟡 | OOF AUC 0.7704862 |
| 2 | Ensemble (`src/ensemble_lgbm_catboost.py`) | 🟡 | Best blend 0.55/0.45, OOF AUC 0.7725910 |
| 2 | Held-out test scoring (`src/evaluate_on_test.py`) | 🟡 | Real run: LGBM 0.7722, CatBoost 0.7700, blend 0.7720 (fold-averaged predictions, see KNOWN_ISSUES #3) |
| 2 | Other experiments (`gbm_5fold_cv.py`, `compare_lightgbm_strategies.py`) | 🟡 | Result files committed |
| 2 | **Final model saved + model card** | 🧪 / ⬜ | Draft exists only in the sandbox. **No trained model for FE v2 is saved anywhere** |
| 2 | **SHAP / LIME** (`src/explainability.py`) | ⬜ | Not started |
| 2 | Select best model on accuracy + explainability-consistency (O5) | ⬜ | Explainability half not done |
| 3 | Counterfactual engine | ⬜ | Only the plan's scaffold exists in the planning document |
| 3 | Fairness audit | ⬜ | Same |
| 4 | Router, loan terms, API (`api/`), frontend (`frontend/`), PDF, DB | ⬜ | Directories listed in the README do not exist yet |
| 5 | Evaluation, manuscript, README rewrite, CI | ⬜ | README is stale |

Other items:

| Item | Status |
|---|---|
| Copilot review on PR #3: High (missing `models/`) | ✅ fixed (`os.makedirs` in `scale_and_balance.py` and `train_baseline.py`) with test |
| Copilot: SMOTE interpolates one-hot indicators | ✅ mitigated: SMOTE off by default, `--smote` uses SMOTENC, test added |
| Copilot Medium: outputs not consumed by training or backend | ⬜ **Open** |
| Copilot Medium: stage disconnected from the documented pipeline | ⬜ **Open** |
| Tests provided in chat but not confirmed added or run: `tests/test_data_prep.py` (in-memory version), `src/thresholds.py` + `tests/test_thresholds.py` | ❓ (the latter is probably obsolete after the three-band decision) |

---

## 4. Last completed work

- **Last repo-changing action by the user:** merged PR #4 (experiments), synced `main`, then ran `git checkout dev-varun && git merge --ff-only main && git push` (output shown: fast-forward `69b13b9..3cf328d`).
- **Last results obtained on real data:** `python3 src/evaluate_on_test.py` →

  | model | test_roc_auc | test_pr_auc | test_logloss |
  |---|---|---|---|
  | lgbm_fe_v2 | 0.7722 | 0.2631 | 0.2417 |
  | catboost_fe_v2 | 0.7700 | 0.2599 | 0.2424 |
  | blend_w0.55 | 0.7720 | 0.2627 | 0.2418 |

- **Last facts checked about gender (real data):** default rate for `CODE_GENDER_M=True` is 0.101297 (n=83,971) against 0.070070 (n=162,037) for False. `CODE_GENDER_M` has gain importance 4219.66 (1.02%) in the FE v2 LightGBM. `feature_engineering_v2.py` contains no reference to `CODE_GENDER`, `NAME_FAMILY_STATUS` or `CNT_CHILDREN`.
- **Last assistant work:** built Phase 1 ("Lock the model") in the assistant's sandbox on **synthetic mock data**, ran 24 tests there (all passed, with `MIN_TEST_AUC=0.5`), and was then told by the user that this was not the requested format (see section 6).

---

## 5. Currently active phase and task

**Active work item:** assistant-proposed Phase 1, "Lock the model" (Part C of `MASTER_PLAN.md`). Under the original roadmap this is the "best model selected" deliverable of Phase 2.

**Task status:** components written and sandbox-checked 🧪; **not delivered** to the user in the requested chat format; nothing applied to the repo.

Components (sandbox draft, to be re-presented in chat):

| File to create | Purpose |
|---|---|
| `src/model_params.py` | `PARAMS` dict extracted from `params` in `src/lgbm_fe_v2_cv.py` (the values are known; see TECHNICAL_DECISIONS §4) |
| `src/decision.py` | `decide()` and `decide_one()`: three-band router (approve below `approve_below`, reject from `reject_from`, review between; validates thresholds and probabilities) |
| `src/final_model.py` | Trains LightGBM on FE v2 features with gender excluded; early stopping on a validation split; picks both cutoffs on validation only; scores test once; saves `models/final_lgbm.pkl`, `models/final_model_card.json`, `reports/final_model_bands.csv`, `data/processed/audit_test.csv`; optional `--ablation` writes `reports/gender_ablation.csv` |
| `tests/conftest.py` | Puts `src/` on the import path |
| `tests/test_decision.py` | Router unit tests (no data needed) |
| `tests/test_fe_v2.py` | FE v2: reproduces the saved feature file; one row equals batch; row-order independent; no infinities; no dependence on gender. Requires FE v2 to be importable |
| `tests/test_final_model.py` | Gender not a feature; model and card agree; card AUC equals a fresh score; AUC floor; scores are probabilities; thresholds ordered; risk rises across bands; approve band keeps the target recall on unseen data |
| One-off edit | Wrap the module-level code of `src/feature_engineering_v2.py` (from the line `X_train = pd.read_csv(TRAIN_X)`, about line 619) in `if __name__ == "__main__":` so that `add_features` can be imported |

Sandbox-only results (synthetic data; **never cite as project results**): 24 tests passed. The draft first failed because the reject cutoff came out below the approve cutoff, so a cap `MAX_REJECT_SHARE = 0.10` was added.

**Corrections the draft still needs before it is given to the user** (found while writing these documents):
1. **Drop `DAYS_BIRTH`.** The CV scripts drop it; the draft's feature list does not. Without this the final model does not match the 0.7716 setup. Add a test that it is absent.
2. **Early stopping 150 rounds** (the CV script uses 150; the draft uses 100).
3. **Parameters are already known** (from `lgbm_fe_v2_cv.py`, TECHNICAL_DECISIONS §4). The assistant had asked the user for `sed` output it did not need.
4. Decide how to report test AUC (KNOWN_ISSUES #3): a single 80%-fit model, a 5-fold average, or a full-train refit.
5. Cutoff targets (recall 0.70 for the approve band, precision 0.30 and at most 10% share for the reject band) are proposals, not measured on real data.

---

## 6. Working agreements (the user was explicit about these)

- **Format:** paste code **in the chat**, say exactly **where it goes**, then give the **tests to run**, then the **commit commands**. The user rebuked the assistant for building files and running long tool sessions instead. Create files only when the user asks (the five documents in this package were asked for).
- **Phase by phase, with a test gate after each phase.** The user's reference style is another project (Aurum): small phases, numbered tasks, decision IDs, "expected output", "please send back" lists.
- **One thing at a time.** The user complained: "Again we are mixing up everything". Do not stack unrelated checks or tasks in one message.
- **Fast and decisive:** "lets just try to do it as fast as I can"; "the rest I leave to you".
- **Code comments:** keep them minimal and natural, so the code does not look machine-generated.
- **Git hygiene:** add files **by name**, never `git add .`; read `git status` before committing; work on `dev-varun`; do not commit data, models, `.env`, or per-applicant prediction dumps.
- **Be honest about unverified claims.** The assistant made mistakes the user noticed (see IMPLEMENTATION_LOG §9). Verify before asserting.
- **Platform:** the user is currently on the Mac (zsh). `pip3` and `python3`. Older instructions were written for PowerShell.
- **Security:** a Kaggle API token was pasted into the conversation. It is deliberately **not** recorded here. The user should revoke and regenerate it.

---

## 7. Exact next task and open questions

### Exact next implementation task

**Deliver Phase 1 "Lock the model" to the user in chat format**, covering:

1. **1.1** Sync `dev-varun` with `main` (`git checkout dev-varun && git merge --ff-only main`), and clean up the stale files in section 2 (including staging the `backend/models/model.py` deletion).
2. **1.2** Create `src/model_params.py` from the `params` dict in `src/lgbm_fe_v2_cv.py`.
3. **1.3** Wrap `src/feature_engineering_v2.py`'s module-level code in a main guard.
4. **1.4** Add `tests/conftest.py`, `src/decision.py`, `src/final_model.py` (with the corrections above), and the three test files.
5. **1.5** Run the unit tests (`test_decision.py`, `test_fe_v2.py`).
6. **1.6** Run `python3 src/final_model.py --ablation` on the real data and report AUCs, cutoffs and the band table.
7. **1.7** Run `tests/test_final_model.py`.
8. **1.8** Commit (by file name) and push to `dev-varun`.

### Questions the user must answer (not yet answered)

| # | Question | Why it matters |
|---|---|---|
| Q1 | Confirm that gender is **excluded from the model itself** (not just hidden from explanations) and kept only for the fairness audit | B5 was ambiguous |
| Q2 | How should the final model be reported: single 80%-fit LightGBM, 5-fold average, or full-train refit? | The 0.7722 test AUC is a fold average |
| Q3 | Real review and submission dates | Never provided. The plan's timeline section is empty |
| Q4 | What have Members 2, 3 and 4 already built? | Unknown. Risk of duplicated work (this already happened once with preprocessing) |
| Q5 | Age and family status: keep as features and audit, or exclude like gender? | Affects fairness and recourse |
| Q6 | Is "phase 4.1, 4.2" a numbering scheme for sub-steps? Is the 8-phase breakdown accepted? | Part C is unratified |
| Q7 | Registration page: seeded demo applicants only, or real sign-up? | Affects the applicant portal |
