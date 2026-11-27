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
| Actionable recourse for applicants who are not approved | ❌ | ✅ Robust, plausible, two-tier recourse (Reject → Review → Approve), including waiting options |
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
docs/           project state, implementation log, technical decisions, known issues, plan
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

**Model.**
- LightGBM on 300 engineered features, with gender excluded: test ROC-AUC 0.7668, PR-AUC 0.2553.
- Excluding gender costs 0.0017 AUC.
- Bands: Approve 34.8% of applicants (actual default rate 2.1%), Review 45.0% (6.7%), Reject 20.1% (21.6%).

**Baselines** (80/20 split, SMOTE on training data as in the base paper; precision, recall and F1 at each model's F1-optimal threshold):

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.722 | 0.217 | 0.407 | 0.283 |
| Random Forest | 0.703 | 0.185 | 0.401 | 0.253 |
| XGBoost | 0.747 | 0.225 | 0.435 | 0.296 |
| LightGBM | 0.748 | 0.237 | 0.404 | 0.299 |

**Explanations** (150 applicants, `reports/explanation_consistency.json`). SHAP additivity is exact. Local LIME against SHAP:
- rank correlation 0.64;
- sign agreement on SHAP's top 5 is 74%;
- seed stability 0.83.

The previous whole-model LIME set-up scored 0.00 on rank correlation and gave nearly the same explanation for everyone.

**Recourse** (300 rejected and 300 Review applicants, `reports/recourse_evaluation.json`). An option counts as usable when it is plausible and clears a safety margin.

| | Usable option | Mean change | Levers changed |
|---|---|---|---|
| Rejected → Review, proposed with waiting | **67.0%** | 0.255 | 1.77 |
| Rejected → Review, proposed (loan levers) | 59.0% | 0.193 | 1.57 |
| Rejected → Review, random search | 61.3% | 0.286 | 3.00 |
| Rejected → Review, previous engine | 21.3% | 0.254 | 1.40 |
| Review → Approve, proposed with waiting | **43.7%** | 0.309 | 2.01 |

On the same 20 applicants per tier, DiCE produced a usable option for 5% (Reject) and 20% (Review), against 60% and 50% for the proposed engine.

**Fairness** (offline audit). Age-band demographic parity difference 0.349; children / no children 0.045. Age is a model feature and is audited; the disparity is documented in `docs/KNOWN_ISSUES.md`.

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
