# CreditXplain

**An Explainable, Counterfactual-Driven Decision Support System for Credit Risk Prediction**

B.Tech Final Year Major Project - KMIT, Hyderabad | Academic Year 2026–2027
Target venue: IEEE Access (or comparable Q1/Q2 peer-reviewed venue)

[![CI](https://github.com/MKN-Sai-Varun/CreditXplain/actions/workflows/ci.yml/badge.svg?branch=dev-varun)](https://github.com/MKN-Sai-Varun/CreditXplain/actions/workflows/ci.yml)
[![Phase](https://img.shields.io/badge/phase-5%20of%205-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## Overview

CreditXplain extends an explainable credit-scoring baseline, Shreya & Pathak (2025),
[arXiv:2506.19383](https://arxiv.org/abs/2506.19383), with three capabilities the base paper does not provide:

| Capability | Base paper | CreditXplain |
|---|---|---|
| Risk prediction + SHAP/LIME explanations | ✅ | ✅ LightGBM, test ROC-AUC 0.767 (base paper 0.720); local LIME checked against SHAP |
| Actionable recourse for applicants who are not approved | ❌ | ✅ Plausible, two-tier recourse (Reject → Review → Approve), including waiting options, with advice that survives model retraining listed first |
| Fairness audit across subgroups | ❌ | ✅ Offline audit plus live monitoring with alerts |
| Interactive decision-support tool | ❌ static HTML/JSON/PNG | ✅ Applicant portal, officer dashboard, PDF reports |

The application follows a three-band decision router:
- **Approve** below a low risk cutoff: loan terms are priced at once.
- **Reject** from a high cutoff: the applicant gets recourse options.
- **Review** in between: a loan officer decides, with a written justification required to override the model.

## Architecture

```
applicant portal ─┐                                  ┌─ PDF reports (decision, compliance)
(Next.js)         ├─ FastAPI ─ inference pipeline ───┤
officer dashboard ┘    │       (FE v2 → LightGBM)    ├─ SHAP + local LIME
                       │                             ├─ recourse engine (src/recourse.py)
                    SQLite                           └─ fairness monitor (Fairlearn)
```

## Repository structure

```
src/            ML pipeline: data prep, feature engineering, final model, inference,
                explanations (SHAP, local LIME), recourse engine, fairness audit, evaluations
backend/        FastAPI app: auth, applications, predictions, recourse, documents,
                officer workflow, fairness monitoring, PDF reports (see backend/README.md)
frontend/       Next.js 16 app: applicant portal and officer dashboard (see frontend/README.md)
tests/          model and pipeline tests
backend/tests/  API tests
scripts/        run_pipeline.sh: rebuilds every artifact and report from the raw data
reports/        results: model comparison, explanations, recourse and fairness evaluations
paper/          manuscript source (IEEE Access); tables and figures come from reports/paper/
docs/           project state, implementation log, technical decisions, known issues, plan,
                usability materials (expert review, accessibility audit)
data/, models/  local only (not in git): raw data, processed data, trained model
```

## Setup

Requirements: Python 3.9, Node.js 20 or newer.

```bash
git clone https://github.com/MKN-Sai-Varun/CreditXplain.git
cd CreditXplain
python3 -m pip install -r requirements.txt
(cd frontend && npm ci)
```

### Data

The project uses the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset. Accept the competition rules on Kaggle, set up the Kaggle CLI (`pip install kaggle` and an API token), then:

```bash
kaggle competitions download -c home-credit-default-risk
unzip home-credit-default-risk.zip -d data/home_credit/
```

The pipeline reads `application_train.csv`, `bureau.csv` and `previous_application.csv`.

### Rebuild the model and every report

```bash
scripts/run_pipeline.sh          # all steps
scripts/run_pipeline.sh --list   # show the steps
scripts/run_pipeline.sh 7        # resume from a step (11 steps)
```

| Step | Script | Produces |
|---|---|---|
| 1 | `src/data_prep.py` | `data/processed/app_train_clean.csv` (307,511 rows, 8.07% default rate) |
| 2 | `src/train_baseline.py` | stratified 80/20 split, `reports/baseline_model_comparison.csv` |
| 3 | `src/feature_engineering_v2.py` | `X_train_fe_v2.csv`, `X_test_fe_v2.csv` |
| 4 | `src/final_model.py --ablation` | `models/final_lgbm.pkl`, `models/final_model_card.json`, `reports/final_model_bands.csv`, `reports/gender_ablation.csv` |
| 5 | `src/build_feature_defaults.py` | `models/feature_defaults.json` |
| 6 | `src.build_recourse_reference` | `models/recourse_reference.npz` |
| 7 | `src/explainability.py` | `reports/shap/`, `reports/lime/`, `reports/explainability_report.md` |
| 8 | `src.explanation_eval` | `reports/explanation_consistency.json` |
| 9 | `src/fairness_audit.py` | `reports/fairness/` |
| 10 | `src.train_bootstrap_models --n 10` | 10 retrained models (`models/bootstrap_lgbm_*.pkl`: 1–5 used by the robust search, 6–10 only for evaluation), `reports/bootstrap_models.json` |
| 11 | `src.recourse_eval --robustness` | `reports/recourse_evaluation.json` |

Dataset description: `python3 -m src.eda` writes `reports/eda/` (report, figures, summary). Paper material: `python3 -m src.paper_figures` builds every table and figure of the paper into `reports/paper/` (see its README). System measurements, run after the pipeline: `python3 -m src.system_latency` (speed), `python3 -m src.time_cost_sensitivity` (price of waiting) and `python3 -m src.understandability_eval` (readability and plain wording), each writing a JSON file to `reports/`.

Experiments behind the model choice (cross-validation, CatBoost, the blend, threshold strategies) are in `src/lgbm_fe_v2_cv.py`, `src/catboost_fe_v2_cv.py`, `src/ensemble_lgbm_catboost.py`, `src/evaluate_on_test.py`, `src/compare_lightgbm_strategies.py` and `src/scale_and_balance.py` (scaling and SMOTENC variants; the final model does not use their outputs).

## Run the application

```bash
python3 -m backend.seed_db                                    # demo users, once
python3 -m uvicorn backend.main:app --reload --port 8000      # from the project root
(cd frontend && npm run dev)                                  # in a second terminal
```

Open http://localhost:3000. Demo accounts (password `password123`):
- `applicant@creditxplain.com`
- `officer@creditxplain.com`

New applicants can also register. The application form has sample profiles that land in each decision band. API documentation is at http://localhost:8000/docs.

## Tests

```bash
python3 -m pytest tests backend/tests -q
(cd frontend && npm run lint && npm run build && npx tsc --noEmit)
```

Tests that need the trained model or processed data skip when those files are absent (as in CI). GitHub Actions runs both suites on every push (`.github/workflows/ci.yml`).

## Results (held-out test set, 61,503 applicants)

**Data** (`reports/eda/`): 307,511 Home Credit applicants with 122 columns and an 8.07% default rate. 41 columns are missing for more than half of the applicants. The credit-bureau scores are the strongest single signals, and the default rate falls with age, from 12.3% under 25 to 5.2% over 55.

**Model.**
- LightGBM on 300 engineered features, with gender excluded: test ROC-AUC 0.7668, PR-AUC 0.2549.
- Excluding gender costs 0.0015 AUC.
- Bands: Approve 34.9% of applicants (actual default rate 2.1%), Review 44.9% (6.7%), Reject 20.2% (21.4%).
- Every fill-in value and clipping limit used in cleaning is learned from the training rows only (`src/data_prep.py`).

**Baselines** (80/20 split, SMOTE on training data as in the base paper; precision, recall and F1 at each model's F1-optimal threshold):

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.722 | 0.220 | 0.395 | 0.283 |
| Random Forest | 0.701 | 0.184 | 0.394 | 0.251 |
| XGBoost | 0.748 | 0.225 | 0.428 | 0.295 |
| LightGBM | 0.747 | 0.229 | 0.424 | 0.297 |

**Explanations** (150 applicants, `reports/explanation_consistency.json`). SHAP additivity is exact. Local LIME against SHAP:
- rank correlation 0.64;
- sign agreement on SHAP's top 5 is 73%;
- seed stability 0.83.

The previous whole-model LIME set-up scored 0.00 on rank correlation and gave nearly the same explanation for everyone.

**Recourse** (300 rejected and 300 Review applicants, `reports/recourse_evaluation.json`). An option is *usable* when it is plausible and clears a safety margin. *Survives retraining* means it still reaches the promised band on 5 bootstrap-retrained models that the search never saw.

| Rejected → Review | Usable option | Usable and survives retraining | Mean change | Levers changed |
|---|---|---|---|---|
| **Hybrid (what the application shows)** | **62.0%** | **23.7%** | 0.307 | 1.91 |
| Proposed with waiting, no retraining check | 62.0% | 14.3% | 0.217 | 1.58 |
| Proposed, loan levers only | 57.3% | 14.0% | 0.185 | 1.48 |
| Random search | 59.7% | 17.7% | 0.269 | 3.00 |
| Previous engine | 22.7% | 8.7% | 0.255 | 1.38 |

For Review → Approve, the hybrid gives 45.3% of applicants a usable option, and 11.0% one that survives retraining. On the same 20 applicants per tier, DiCE produced a usable option for 20% (Reject) and 5% (Review), against 60% and 40% for the hybrid.

The hybrid lists first the options that all 5 retrained search models agree on, and labels each option as likely or not to hold if the model is updated. Requiring that agreement roughly doubles how often an option survives retraining (24.4% → 53.3%), at some cost in coverage. The hybrid keeps full coverage by falling back to the standard options.

**System performance** (`reports/system_latency.json`, 100 applicants, laptop):
- An applicant's full path (submit, decision with SHAP, LIME and recourse, PDF report) takes 158 ms on average and 259 ms at the 95th percentile.
- Recourse is the largest stage (95 ms on average, Review and Reject only), then LIME (38 ms).
- Start-up takes 1.6 s.
- The advice is insensitive to the price put on waiting: doubling it changes the first suggestion for at most 4% of applicants, halving it for 6–10%.

**Understandability** (`reports/understandability.json`, 150 applicants, 50 per decision):
- The text an applicant reads scores 60.0 on Flesch reading ease (plain English) and is at about an 8th-grade level.
- It contains no technical terms or raw column names, and every factor shown has a plain-language name.
- An applicant sees five reasons and, if not approved, usually one or two options of about two changes each (median 115 words in all).
- No user study with applicants was run. An expert review is prepared in `docs/usability/`.

**Fairness** (offline audit). Age-band demographic parity difference 0.356; children / no children 0.049.
- Age policy is applied as **eligibility rules before the model**: applicants under 18 apply with a guardian, who is assessed, and an officer decides once the guardian's documents are uploaded. Without an income, no application is approved automatically.
- Removing age from the model was tested (`reports/age_fairness_experiment.json`, run before the cleaning fix of #86, with the model then at DPD 0.349) and **widens** the gap to 0.377, with AUC 0.7646. Age is recoverable from other features (R² 0.735), and the age-free model over-predicts risk for under-25s.
- The current model is calibrated within 0.9 points in every age band. The approval gap follows actual default rates (11.4% for under-25s, 5.3% for over-55s).

## Roadmap

| Phase | Focus | Status |
|---|---|---|
| 1 | Literature review, data acquisition, cleaning | ✅ |
| 2 | Baselines, final model, SHAP/LIME | ✅ |
| 3 | Recourse engine, fairness audit | ✅ |
| 4 | FastAPI backend, PDF reports, Next.js portal and dashboard | ✅ |
| 5 | Evaluation, reproducibility, manuscript | In progress |

Details: `docs/PROJECT_STATE.md`, `docs/IMPLEMENTATION_LOG.md`, `docs/TECHNICAL_DECISIONS.md`, `docs/KNOWN_ISSUES.md`.

## Tech stack

**ML:** scikit-learn · LightGBM · XGBoost · CatBoost · SHAP · LIME · DiCE · Fairlearn
**Backend:** FastAPI · SQLAlchemy · SQLite · ReportLab
**Frontend:** Next.js 16 · React 19 · TypeScript · Tailwind CSS · Recharts

## Team

| Member | Responsibility |
|---|---|
| Varun | Data preprocessing, EDA, baseline model implementation |
| Member 2 | SHAP/LIME explainability integration |
| Member 3 | Counterfactual Recommendation Engine |
| Member 4 | Dashboard, fairness analysis, PDF reporting, manuscript coordination |

## Citation

Base paper this project extends:

```bibtex
@article{shreya2025explainable,
  title={Explainable Artificial Intelligence Credit Risk Assessment using Machine Learning},
  author={Shreya and Pathak, Harsh},
  journal={arXiv preprint arXiv:2506.19383},
  year={2025}
}
```

## License

MIT - see `LICENSE`.

## Note on results

Every number above comes from a file in `reports/` or `models/final_model_card.json`, produced by `scripts/run_pipeline.sh`. Interest rates in the application are illustrative. Recourse suggestions are model-based estimates, not causal guarantees.
