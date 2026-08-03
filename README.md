# CreditXplain

**An Explainable, Counterfactual-Driven Decision Support System for Credit Risk Prediction**

B.Tech Final Year Major Project - KMIT, Hyderabad | Academic Year 2026–2027
Target venue: IEEE Access (or comparable Q1/Q2 peer-reviewed venue)

[![Status](https://img.shields.io/badge/status-in%20development-yellow)]()
[![Phase](https://img.shields.io/badge/phase-1%20of%205-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## Overview

CreditXplain extends a peer-reviewed explainable credit-scoring baseline -
Shreya & Pathak (2025), [arXiv:2506.19383](https://arxiv.org/abs/2506.19383) -
by adding three capabilities the base paper does not provide:

| Capability | Base Paper | CreditXplain |
|---|---|---|
| Risk prediction + SHAP/LIME explainability | ✅ | ✅ Retained as baseline |
| Actionable recourse for rejected applicants | ❌ | ✅ Counterfactual Recommendation Engine |
| Fairness/bias audit across subgroups | ❌ | ✅ Explicit subgroup fairness audit |
| Interactive, live decision-support tool | ❌ Static HTML/JSON/PNG | ✅ Interactive dashboard + PDF export |

The system is designed as a three-sided application:

1. **Applicant self-service portal** - submit an application, check status live, and receive concrete, actionable recourse if rejected.
2. **Bank loan officer dashboard** - risk scores, explanations, historical analytics, compliance alerts, PDF report generation.
3. **Backend decision engine + fairness monitor** - prediction, explainability, counterfactual generation, and bias auditing.

## Repository Structure

```
creditxplain/
├── data/
│   ├── home_credit/     # raw Kaggle CSVs (not tracked in git)
│   ├── processed/       # cleaned/engineered output (not tracked)
│   └── uploads/         # applicant document uploads (Phase 4)
├── models/               # saved model artifacts
├── reports/              # EDA plots, SHAP plots, PDF reports
├── src/                  # core pipeline code
├── app/                  # Streamlit dashboard + applicant portal
├── notebooks/            # exploratory analysis
├── tests/                # unit tests
└── requirements.txt
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/creditxplain.git
cd creditxplain
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Obtain the dataset

This project uses the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset (Kaggle), which requires accepting the competition rules before download.

```bash
kaggle competitions download -c home-credit-default-risk
```

Extract into `data/home_credit/`. The pipeline requires:
- `application_train.csv`
- `bureau.csv`
- `previous_application.csv`

(Remaining files are used only for stretch-goal features - see [Dataset Scope](#dataset-scope).)

### 4. Run the data pipeline

```bash
python src/data_prep.py --data-path data/home_credit/ --out data/processed/app_train_clean.csv
```

Produces a cleaned, feature-engineered dataset (307,511 rows × 133 columns) ready for model training.

## Dataset Scope

To keep the project achievable within a 24-week semester timeline, only the
merge/aggregation logic needed to reproduce the base paper's reported top
features is implemented: `EXT_SOURCE_1/2/3`, `CREDIT_TO_GOODS_RATIO`,
`CNT_FAM_MEMBERS`, `AGE_YEARS`. These come primarily from
`application_{train,test}.csv`, with shallow aggregates from `bureau.csv` and
`previous_application.csv`. The remaining relational tables
(`bureau_balance.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`,
`installments_payments.csv`) are treated as stretch-goal feature sources.

## Roadmap

| Phase | Weeks | Focus | Status |
|---|---|---|---|
| 1 | 1–4 | Literature review, dataset acquisition, cleaning, EDA | ✅ Complete |
| 2 | 5–9 | Baseline models (LR, RF, XGBoost, LightGBM) + SHAP/LIME | 🔲 In progress |
| 3 | 10–13 | Counterfactual Recommendation Engine + fairness audit | 🔲 Planned |
| 4 | 14–18 | Interactive dashboard + applicant portal | 🔲 Planned |
| 5 | 19–24 | Evaluation, manuscript, code release | 🔲 Planned |

## Tech Stack

`scikit-learn` · `XGBoost` · `LightGBM` · `SHAP` · `LIME` · `DiCE` · `Fairlearn` · `Streamlit` · `ReportLab`

## Team

| Member | Responsibility |
|---|---|
| Member 1 | Data preprocessing, EDA, baseline model implementation |
| Member 2 | SHAP/LIME explainability integration |
| Member 3 | Counterfactual Recommendation Engine |
| Member 4 | Dashboard, fairness analysis, PDF reporting, manuscript coordination |

## Citation

If referencing the base paper this project extends:

```bibtex
@article{shreya2025explainable,
  title={Explainable Artificial Intelligence Credit Risk Assessment using Machine Learning},
  author={Shreya and Pathak, Harsh},
  journal={arXiv preprint arXiv:2506.19383},
  year={2025}
}
```

## License

MIT - see `LICENSE` for details. *(Add a `LICENSE` file before publishing publicly.)*

## Status Note

This is an active research project. Performance figures cited in project
planning documents are hypothesized targets or numbers reported by the base
paper - not this project's own results - until Phase 2 evaluation is complete.
