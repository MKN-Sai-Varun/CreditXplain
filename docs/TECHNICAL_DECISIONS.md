# TECHNICAL_DECISIONS — CreditXplain

**Last Updated:** Phase 5.2b ensemble-robust recourse (Nov 27, 2026)

This document records all technical and architectural decisions made during the project, with rationale.

---

## 1. Tech Stack

| Component | Technology | Rationale |
|---|---|---|
| ML Framework | scikit-learn, LightGBM, CatBoost, XGBoost | Industry standard, LightGBM selected for performance |
| Explainability | SHAP, LIME | Base paper requirement |
| Counterfactual | Bounded lever search (`src/recourse.py`); DiCE kept offline in `src/counterfactuals.py` | DiCE on engineered columns gave inconsistent ratios and low success; see D-38 |
| Fairness | Fairlearn | Microsoft-backed, well-documented |
| Backend | FastAPI | Modern async framework, auto-docs, type hints |
| Frontend | React + Next.js 16 (App Router, TypeScript, Tailwind 4) | Changed from Streamlit (Amendment B1); see D-61 |
| Database | SQLite | Sufficient for single-user design, easy deployment |
| PDF Generation | ReportLab (planned, not implemented yet) | Python-native, template support |
| Testing | pytest | Standard Python testing framework |

---

## 2. Architecture Decisions

### D-1: Final Model Selection (Phase 1)
**Decision:** Single LightGBM on Feature Engineering V2 features  
**Alternatives Considered:**
- LightGBM + CatBoost ensemble (55/45 blend)
- CatBoost alone
- Baseline LightGBM

**Rationale:**
- LightGBM + FE V2 achieved **highest test ROC-AUC: 0.7722**
- Ensemble gave only +0.001 in validation but *lost* 0.0002 on test (0.7720 vs 0.7722)
- Single model simpler to deploy, interpret, and explain
- Better generalization (test exceeded validation)

**Date:** Nov 19, 2026  
**Evidence:** `reports/fe_v2_test_results.csv`, `reports/lgbm_fe_v2_results.csv`

---

### D-2: Three-Band Decision Router (Phase 1)
**Decision:** Approve / Review / Reject bands instead of binary threshold  
**Alternatives Considered:**
- Single threshold (default 0.5)
- Single F1-optimized threshold
- Dual thresholds with binary decision

**Rationale:**
- Matches workflow diagram's "Decision Router" component
- Handles real-world review queue requirement
- Allows different risk tolerance per band
- Settles the 0.08 vs 0.162 threshold debate by using both

**Thresholds (selected on validation set):**
- **Approve below:** Target 70% recall for minority class
- **Reject from:** Target 30% precision, capped at 10% of population
- **Review between:** Manual officer review

**Date:** Nov 19, 2026  
**Implementation:** `src/decision.py`

---

### D-3: Gender Exclusion Policy (Phase 1)
**Decision:** **Exclude CODE_GENDER entirely from the model**  
**Alternatives Considered:**
- Keep gender in model, hide in explanations
- Keep gender, audit only
- Exclude gender and age/family status

**Rationale:**
- Gender is **not a valid credit decision factor** (discriminatory)
- Age **is valid** (legal capacity, guardian requirements for minors)
- Family status **is valid** (co-applicant eligibility, repayment capacity)
- Gender will be saved separately for **fairness audit in Phase 3**

**Date:** Nov 19, 2026  
**Confirmed by project lead:** "Gender should not be taken into consideration by the model"  
**Implementation:** `src/model_params.py` (GENDER_PATTERNS exclusion list)

---

### D-4: Evaluation Protocol (Phase 1)
**Decision:** Train / Validation / Test split with proper isolation  
**Train (80% of training data):** Model fitting  
**Validation (20% of training data):** Early stopping + threshold selection  
**Test (original 20% split):** Reported once, never tuned on

**Rationale:**
- Previous baseline models chose thresholds on test set (leakage)
- Validation set prevents threshold overfitting
- Matches standard ML practice

**Date:** Nov 19, 2026  
**Implementation:** `src/final_model.py` (train_test_split with 0.2 validation)

---

## 3. Feature Engineering Decisions

### FE-1: DAYS_BIRTH Exclusion
**Decision:** Drop DAYS_BIRTH, use AGE_YEARS (derived: -DAYS_BIRTH / 365)  
**Rationale:**
- DAYS_BIRTH is negative days from birth (confusing)
- AGE_YEARS is human-interpretable
- Redundant features after transformation

**Date:** Phase 2 (Feature Engineering V2)  
**Implementation:** `src/feature_engineering_v2.py`, exclusion in `src/model_params.py`

---

### FE-2: Feature Engineering V2 Strategy
**Decision:** 15 feature groups, 100+ derived features  
**Groups:**
1. External source aggregates (mean, min, max, std, interactions)
2. Financial burden ratios (credit/income, annuity/income, etc.)
3. Age/employment features (employment age ratio, tenure)
4. Credit/income + external source interactions
5. Bureau features (bureau credit ratios, overdue ratios)
6. Previous application features (refusal rate, approval count)
7. Social circle features (default rates at 30/60 days)
8. Credit bureau request aggregates
9. Document counts
10. Contact method counts
11. Region risk features
12. Region/city discrepancies
13. Property quality aggregates
14. Log transforms of amount columns
15. Missingness indicators

**Rationale:**
- Captures non-linear relationships
- Domain-informed interactions
- Improved validation AUC from 0.748 (baseline) to 0.7716 (FE v2)

**Date:** Phase 2  
**Implementation:** `src/feature_engineering_v2.py`

---

## 4. LightGBM Hyperparameters (Final Model)

**Source:** Extracted from `src/lgbm_fe_v2_cv.py` (best-performing CV run)

```python
{
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 35,
    "max_depth": 6,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "random_state": 42
}
```

**Early Stopping:** 150 rounds  
**Rationale:** Tuned during CV experiments; prevents overfitting

**Date:** Phase 2  
**Validation Performance:** 0.7716 OOF AUC, 0.7722 Test AUC

---

## 5. SMOTE Decision (Phase 2)

**Decision:** SMOTE **disabled by default**, optional via `--smote` flag  
**Rationale:**
- Validation experiments showed SMOTE lowered ROC-AUC
- Class imbalance handled via decision thresholds (cheaper, more flexible)
- When used, SMOTENC employed (handles binary flags correctly)

**Date:** Phase 2  
**Implementation:** `src/scale_and_balance.py`  
**Evidence:** The PR #3 code review caught plain SMOTE interpolating one-hot indicators

---

## 6. Data Cleaning Pipeline (Phase 1)

### Outlier Handling
**Method:** 3σ clipping (clip to mean ± 3 standard deviations)  
**Exclusions:** ID columns, TARGET, binary flags (0/1 only)  
**Rationale:** Preserves distribution shape, removes extreme outliers

### Missing Value Imputation
**Numeric:** Median imputation  
**Categorical:** Mode imputation  
**Special Case:** DAYS_EMPLOYED == 365243 flagged as anomaly, then nulled

### Bureau and Previous Application Aggregation
**Strategy:** Aggregate to one row per SK_ID_CURR **before** merging  
**Rationale:** Prevents row duplication in main application table

**Date:** Phase 1  
**Implementation:** `src/data_prep.py`  
**Verified Output:** 307,511 rows × 133 columns, 8.07% default rate, 0 NaNs

---

## 7. Test Strategy

### Unit Tests (Fast)
- **decision.py:** Router logic, threshold validation, edge cases
- **feature_engineering_v2.py:** Importability, feature creation, determinism

### Integration Tests (Requires Trained Model)
- **final_model.py:** Gender exclusion, threshold ordering, AUC floor, model card completeness

**Minimum AUC Threshold:** 0.70 (test fails below this)

**Date:** Phase 1  
**Location:** `tests/`

---

## 8. Git Workflow

**Development Branch:** `dev-varun`  
**Protected Branch:** `main`  
**Merge Strategy:** Pull requests only (no direct commits to main)  
**Commit Hygiene:** Stage files by name (`git add <file>`), never `git add .`

**Gitignored:**
- `data/` (raw and processed datasets)
- `models/` (trained model artifacts)
- `.env` (secrets)
- Per-applicant prediction dumps

**Date:** Established Phase 1

---

## 9. Directory Structure

```
creditxplain/
├── data/
│   ├── home_credit/          # Raw Kaggle CSVs (gitignored)
│   ├── processed/            # Cleaned/engineered data (gitignored)
│   └── uploads/              # Applicant document uploads (Phase 4)
├── models/                    # Trained models (gitignored)
├── reports/                   # EDA plots, model comparison CSVs, feature importance
├── src/                       # ML pipeline scripts
├── tests/                     # pytest unit and integration tests
├── docs/                      # Project documentation
├── api/                       # FastAPI backend (Phase 4)
├── frontend/                  # Next.js app (Phase 4)
└── requirements.txt
```

---

## 10. Dependency Management

**Python Version:** 3.9.6 (system Python on macOS)  
**Package Manager:** pip3  
**Virtual Environment:** Not used (system-level installation)

**Key Dependencies:**
- pandas, numpy, scikit-learn
- lightgbm, xgboost, catboost
- imbalanced-learn (SMOTENC)
- shap, lime, dice-ml, fairlearn
- joblib (model persistence)
- pytest (testing)

**Date:** Established Phase 1  
**File:** `requirements.txt`

---

## 11. Decisions Deferred to Later Phases

### DF-1: Test Reporting Method
**Question:** Single 80%-fit model, 5-fold average, or full-train refit?  
**Current:** Single 80%-fit model  
**Note:** 5-fold CV test results (0.7722) are fold-averaged predictions  
**Defer To:** Phase 5 (manuscript preparation)

### DF-2: Age/Family Status in Fairness Audit
**Question:** Audit for disparities or exclude like gender?  
**Current:** Keep in model, audit in Phase 3  
**Rationale:** Age is legally relevant (guardian requirements)  
**Defer To:** Phase 3 (fairness audit implementation)

### DF-3: Threshold Targets
**Current Values:**
- Approve: 70% recall for minority class
- Reject: 30% precision, max 10% population share

**Status:** Selected on validation set, not yet validated on business metrics  
**Defer To:** Phase 4 (business stakeholder review)

---

## 11. Explainability Decisions (Phase 2)

### E-1: SHAP as Primary Explainability Method
**Decision:** SHAP (TreeExplainer) as primary, LIME as secondary validation  
**Rationale:**
- SHAP is exact for tree-based models (no approximation)
- Mathematically grounded (game theory, Shapley values)
- Satisfies additivity property (base + sum = prediction)
- Deterministic (perfect stability across runs)
- Provides both global and local explanations

**LIME Role:** Secondary validation and stakeholder communication
- Simpler for non-technical audiences
- HTML outputs easier to share
- 58.3% agreement with SHAP validates consistency

**Date:** Nov 19, 2026  
**Implementation:** `src/explainability.py`  
**Evidence:** `reports/explainability_report.md`

---

### E-2: Sample Size for SHAP Computation
**Decision:** 5,000 samples (default) with configurable flag  
**Alternatives:**
- Full test set (61,503) - accurate but slow (~15 min)
- Smaller sample (1,000) - fast but less representative

**Rationale:**
- 5,000 samples balances accuracy and speed (~5 min)
- Statistically representative (8% of test set)
- Includes sufficient samples from all decision bands
- Configurable via `--sample-size` flag for flexibility

**Date:** Nov 19, 2026

---

### E-3: Top Features Identified

Based on SHAP global importance analysis (5,000 test samples):

| Rank | Feature | SHAP Importance | Category |
|---|---|---|---|
| 1 | CREDIT_GOODS_EXT_RISK | 0.2493 | Engineered interaction |
| 2 | EXT_SOURCE_MEAN | 0.2305 | External credit score |
| 3 | CREDIT_ANNUITY_RATIO | 0.1323 | Financial burden |
| 4 | Higher Education | 0.0845 | Demographics |
| 5 | DAYS_ID_PUBLISH | 0.0647 | Document recency |

**Key Insights:**
- **External credit scores** (EXT_SOURCE_*) dominate predictions
- **Engineered features** (FE V2) add significant value
- **Financial burden ratios** (payment capacity) highly predictive
- **Education level** affects default risk
- **Gender excluded** - not in top features, ethical decision validated

**Actionability (Phase 3 input):**
- Credit ratios → actionable (reduce credit amount, increase income)
- External scores → partially actionable (improve credit history over time)
- Education → not actionable (immutable)
- Document age → somewhat actionable (update documents)

**Date:** Nov 19, 2026  
**Evidence:** `reports/shap/global_feature_importance.csv`

---

## Change Log

| Date | Phase | Changes |
|---|---|---|
| Nov 19, 2026 | Phase 1 Start | Initial document created; D-1 through D-4, FE-1, FE-2 documented |
| Nov 19, 2026 | Phase 1 Complete | D-2 threshold strategy refined (40-40-20 → 35-45-20 actual); D-3 validated via ablation |
| Nov 19, 2026 | Phase 2 Complete | E-1 documented (SHAP primary, LIME secondary); top 5 features identified |

---

**Note:** This document is a living record. Update after each phase with new decisions and their rationale.


---

## 12. Counterfactual & Fairness Decisions (Phase 3)

### CF-1: Counterfactual Library Selection
**Decision:** DiCE (Diverse Counterfactual Explanations) library  
**Alternatives Considered:**
- Custom implementation (gradient-based)
- CARLA framework
- Alibi library

**Rationale:**
- DiCE is established and well-maintained
- Supports multiple ML backends (sklearn-compatible)
- Generates diverse counterfactuals (not just nearest)
- Works with both continuous and categorical features
- Academic foundation (Mothilal et al., 2020)

**Date:** Nov 19, 2026  
**Implementation:** `src/counterfactuals.py`

---

### CF-2: Actionable vs Immutable Features
**Decision:** Define explicit ACTIONABLE_FEATURES list (27 features)  
**Actionable Features:**
- Financial: AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE
- Employment: DAYS_EMPLOYED, EMPLOYMENT_YEARS
- Documents: DAYS_ID_PUBLISH, DAYS_LAST_PHONE_CHANGE, DOCUMENT_COUNT
- Contact: FLAG_WORK_PHONE, FLAG_PHONE, FLAG_EMAIL, FLAG_MOBIL, CONTACT_METHOD_COUNT
- Property: FLAG_OWN_CAR_Y, OWN_CAR_AGE
- Family: CNT_FAM_MEMBERS (co-applicant)
- Derived ratios: CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, etc.

**Immutable/Slow-to-Change (273 features):**
- Age (DAYS_BIRTH, AGE_YEARS)
- Gender (already excluded from model)
- Education level
- External credit scores (EXT_SOURCE_*) - require months to improve
- Historical flags

**Rationale:**
- Recourse must be **realistically achievable** by applicant
- External scores are "slow actionable" - noted but de-prioritized
- Focuses counterfactuals on immediate levers (loan amount, documents)

**Date:** Nov 19, 2026  
**Evidence:** Only 27 of 300 features marked actionable in `ACTIONABLE_FEATURES` constant

---

### CF-3: Counterfactual Success Criteria
**Decision:** Two-tier success (APPROVE vs REVIEW)  
**Target:**
- Primary: Flip from REJECT to **REVIEW** (prob < 0.1189)
- Stretch: Flip from REJECT to **APPROVE** (prob < 0.0349)

**Rationale:**
- Approve threshold (0.0349) is **extremely low** - only 3.5% default risk acceptable
- Very few counterfactuals achieve this (3.3% success rate in testing)
- REVIEW band is more realistic goal - still improves applicant outcome
- Provides **graduated recourse** recommendations

**Results (10 rejected applicants, 3 CFs each):**
- Total generated: 30 counterfactuals
- Successful (REVIEW or better): 1 (3.3%)
- Average changes required: 1.5 features
- Most changed features: INCOME_PER_FAMILY_MEMBER (13%), OWN_CAR_AGE (13%)

**Date:** Nov 19, 2026  
**Evidence:** `reports/counterfactuals/counterfactual_results.csv`

---

### CF-4: Feasibility Scoring
**Decision:** Rank by number of changes (fewer = more feasible)  
**Feasibility Score:** `1.0 / (num_changes + 1)`

**Rationale:**
- Simpler changes are easier for applicants to implement
- Each changed feature adds time, cost, and complexity
- Alternative (distance-based) penalizes large percentage changes, but number is more intuitive

**Date:** Nov 19, 2026

---

### F-1: Fairness Library Selection
**Decision:** Fairlearn (Microsoft) for fairness metrics  
**Alternatives:**
- AIF360 (IBM)
- Custom implementation

**Rationale:**
- Fairlearn is actively maintained, well-documented
- Integrates with scikit-learn ecosystem
- Provides industry-standard metrics (demographic parity, equalized odds)
- Used in production ML systems

**Date:** Nov 19, 2026  
**Implementation:** `src/fairness_audit.py`

---

### F-2: Protected Attributes Selection
**Decision:** Audit **age bands** and **has_children** (family status)  
**Excluded:** Gender (already removed from model)

**Age Bands:**
- <25, 25-35, 35-45, 45-55, 55+

**Family Status:**
- Single, Married, Has Children (simplified from marital status + children count)

**Rationale:**
- Age is **in the model** (legal requirement for guardians) - must audit
- Family status affects loan decisions (co-applicant eligibility) - must audit
- Gender **not in model** - cannot cause model-based disparity (monitoring only)

**Date:** Nov 19, 2026

---

### F-3: Fairness Thresholds
**Decision:** 0.1 threshold for Demographic Parity Difference  
**Standard:** Industry practice (80% rule → 0.2 max, we use stricter 0.1)

**Interpretation:**
- < 0.1: Acceptable disparity
- 0.1 - 0.2: Moderate concern, monitor
- \> 0.2: Significant disparity, requires intervention

**Date:** Nov 19, 2026

---

### F-4: Age Disparity Finding (CRITICAL)
**Finding:** **Significant age-based disparity detected**

**Metrics:**
- Demographic Parity Difference: **0.3488** (threshold: 0.1) ⚠️
- Equalized Odds Difference: **0.3496**

**Approve Rates by Age Band:**
| Age Band | Approve Rate | Default Rate (Actual) |
|---|---|---|
| <25 | 13.36% | 11.37% |
| 25-35 | 24.71% | 10.75% |
| 35-45 | 34.93% | 8.15% |
| 45-55 | 38.76% | 6.83% |
| **55+** | **48.25%** | **5.25%** |

**Disparity:** 34.88 percentage point gap (55+ vs <25)

**Root Cause Analysis:**
1. **Data-driven:** Older applicants have lower actual default rates (5.25% vs 11.37%)
2. **Model learned this pattern** - age is predictive and legally relevant
3. **Not discrimination** - model correlates age with repayment capacity

**Remediation Strategy (Deferred to Phase 4):**
- Monitor for business impact (does disparity harm younger applicants unfairly?)
- Consider **age-stratified thresholds** (different cutoffs per age band)
- Explore **age-blind scoring** as alternative model
- Document in user-facing reports (transparency)

**Date:** Nov 19, 2026  
**Evidence:** `reports/fairness/fairness_report.json`, `reports/fairness/fairness_audit_report.md`

---

### F-5: Family Status (Has Children) - Low Disparity ✓
**Finding:** Acceptable disparity, no intervention needed

**Metrics:**
- Demographic Parity Difference: **0.0447** (well below 0.1 threshold)
- Equalized Odds Difference: **0.0418**

**Approve Rates:**
| Group | Approve Rate | Default Rate |
|---|---|---|
| No Children | 36.18% | 7.68% |
| With Children | 31.71% | 9.00% |

**Disparity:** 4.47 percentage points (acceptable)

**Interpretation:**
- Small gap driven by actual default rate difference (1.32 pp)
- Model behaves fairly with respect to family status
- No remediation required

**Date:** Nov 19, 2026

---

### F-6: Test Strategy - Fairness
**Approach:** Mock-free integration tests with real audit data

**Coverage:**
- Data loading and preprocessing (age bands, family status)
- Fairness metric computation (demographic parity, equalized odds)
- Report generation and structure validation
- Protected attribute handling (gender excluded from model)

**Results:** 22/22 tests passing

**Date:** Nov 19, 2026  
**Location:** `tests/test_fairness.py`

---

### F-7: DiCE Dtype Issue Resolution
**Problem:** DiCE auto-converts low-cardinality features to CategoricalDtype (object), causing errors

**Solution:** Force ALL features as continuous in DiCE Data object:
```python
dice_data = dice_ml.Data(
    dataframe=dice_df,
    continuous_features=list(feature_columns),  # All features, including binary
    outcome_name='outcome'
)
```

**Rationale:**
- Binary features (0/1) can be treated as continuous for counterfactual generation
- Prevents object dtype issues with pandas
- Counterfactuals still make sense (changing flag from 0 to 1)

**Date:** Nov 19, 2026  
**Issue:** Encountered during test execution, fixed in `src/counterfactuals.py`

---

## Change Log

| Date | Phase | Changes |
|---|---|---|
| Nov 19, 2026 | Phase 1 Start | Initial document created; D-1 through D-4, FE-1, FE-2 documented |
| Nov 19, 2026 | Phase 1 Complete | D-2 threshold strategy refined (40-40-20 → 35-45-20 actual); D-3 validated via ablation |
| Nov 19, 2026 | Phase 2 Complete | E-1 documented (SHAP primary, LIME secondary); top 5 features identified |
| Nov 19, 2026 | Phase 3 Complete | CF-1 to CF-4 (counterfactuals), F-1 to F-7 (fairness); age disparity identified (0.3488) |


---

## 13. Backend Architecture Decisions (Phase 4.1)

### D-8: Backend Framework Selection
**Decision:** FastAPI for REST API backend  
**Alternatives Considered:**
- Flask (simpler but less features)
- Django (full framework, overkill for API-only)
- Express.js (different language)

**Rationale:**
- **Async support** - handles concurrent requests efficiently
- **Automatic OpenAPI docs** - interactive API documentation at `/docs`
- **Type hints** - Pydantic integration for request/response validation
- **Modern Python** - uses Python 3.7+ features (async/await)
- **Performance** - comparable to Node.js, faster than Flask
- **Developer experience** - clean syntax, minimal boilerplate

**Date:** Nov 19, 2026  
**Implementation:** `backend/main.py`

---

### D-9: Database Selection
**Decision:** SQLite for application database  
**Alternatives Considered:**
- PostgreSQL (production-grade but requires server)
- MySQL (similar to PostgreSQL)
- MongoDB (NoSQL, overkill for structured data)

**Rationale:**
- **Single-user design** (Amendment B3) - no multi-tenancy needed
- **File-based** - no database server setup required
- **Sufficient for prototype** - handles expected load (< 1000 applications)
- **Easy deployment** - single database file
- **ACID compliant** - supports transactions
- **Migration path** - SQLAlchemy allows easy switch to PostgreSQL later

**Database File:** `creditxplain.db` (backend directory, gitignored)

**Date:** Nov 19, 2026  
**Implementation:** `backend/database.py`

---

### D-10: Database Schema Design

**Decision:** 7-table normalized schema with relationships

**Tables:**
1. **users** - Authentication and roles (applicant/officer/admin)
2. **applications** - Credit applications with JSON features
3. **decisions** - Three-band routing results
4. **explanations** - SHAP/LIME/counterfactuals storage
5. **loan_terms** - EMI and interest rate calculations
6. **audit_logs** - Officer action tracking
7. **fairness_metrics** - Demographic parity monitoring

**Key Design Decisions:**
- **JSON storage for features** - flexible schema, avoids 300+ columns
- **One-to-one** relationships for decision → explanation → loan_terms
- **One-to-many** for user → applications, application → audit_logs
- **Timestamps** on all tables for audit trail
- **Enum types** for status/role (type safety at database level)

**Rationale:**
- Normalized design reduces redundancy
- JSON fields allow flexibility for feature schema evolution
- Enums prevent invalid status values
- Audit trail supports compliance requirements

**Date:** Nov 19, 2026  
**Implementation:** `backend/models.py`

---

### D-11: Authentication Strategy
**Decision:** JWT (JSON Web Tokens) with bcrypt password hashing  
**Alternatives Considered:**
- Session-based auth (requires server-side storage)
- OAuth2 (overkill for prototype)
- API keys (less secure for users)

**JWT Configuration:**
- **Algorithm:** HS256 (HMAC with SHA-256)
- **Expiration:** 24 hours
- **Secret key:** Configurable via environment variable
- **Payload:** User email (sub claim)

**Password Security:**
- **Hashing:** bcrypt with salt
- **Version:** bcrypt 4.0.1 (compatible with passlib 1.7.4)
- **Unique salt** per password (automatic)

**Rationale:**
- JWT is stateless - no server-side session storage
- Industry standard for REST APIs
- Compatible with Next.js frontend
- bcrypt is OWASP-recommended for password hashing

**Date:** Nov 19, 2026  
**Implementation:** `backend/auth.py`  
**Dependencies:** python-jose[cryptography], passlib[bcrypt]

---

### D-12: API Route Organization
**Decision:** Router-based modular organization (5 routers)

**Routers:**
1. **/api/auth** - Authentication (register, login, current user)
2. **/api/applications** - Application CRUD operations
3. **/api/predictions** - ML prediction pipeline (Phase 4.2)
4. **/api/officer** - Officer dashboard (review queue, actions, analytics)
5. **/api/fairness** - Fairness monitoring (metrics, demographics)

**Rationale:**
- **Separation of concerns** - each router handles one domain
- **Scalability** - easy to add new routers
- **Testability** - routers can be tested independently
- **Code organization** - clear structure for team collaboration

**URL Prefix:** All routes under `/api/` for clarity

**Date:** Nov 19, 2026  
**Implementation:** `backend/routers/`

---

### D-13: Request/Response Schema Validation
**Decision:** Pydantic v2 for schema validation  
**Approach:** Separate schema classes for create/response/detail views

**Example Pattern:**
```python
# Request
class ApplicationCreate(BaseModel):
    features: Dict[str, Any]

# Response
class ApplicationResponse(BaseModel):
    id: int
    status: ApplicationStatus
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)
```

**Rationale:**
- **Type safety** - automatic validation of incoming requests
- **Documentation** - schemas appear in OpenAPI docs
- **Separation** - create/update schemas differ from response schemas
- **Flexibility** - response schemas can exclude sensitive fields

**Date:** Nov 19, 2026  
**Implementation:** `backend/schemas.py` (20+ schema classes)

---

### D-14: CORS Configuration
**Decision:** Enable CORS for localhost:3000 and localhost:3001  
**Allowed Origins:**
- http://localhost:3000 (Next.js default)
- http://localhost:3001 (backup port)

**Allowed Methods:** All (GET, POST, PUT, DELETE)  
**Allowed Headers:** All  
**Credentials:** Enabled (for JWT cookies if needed)

**Rationale:**
- Frontend and backend on different ports (cross-origin)
- Localhost-only for development security
- Production deployment will use same-origin or specific domains

**Date:** Nov 19, 2026  
**Implementation:** `backend/main.py` (CORS middleware)

---

### D-15: Configuration Management
**Decision:** Pydantic Settings with .env file support  
**Configuration Categories:**
- Application (name, version, debug mode)
- Security (secret key, algorithm, token expiration)
- Database (connection URL)
- Paths (models, data, reports directories)
- Decision thresholds (from Phase 3)
- Loan calculation (base rate, risk spreads)
- CORS (allowed origins)

**Environment File:** `.env` (gitignored, `.env.example` provided)

**Rationale:**
- **12-factor app** principles (configuration via environment)
- **Type-safe** configuration with Pydantic
- **Easy deployment** - different configs per environment
- **Security** - secrets not hardcoded

**Date:** Nov 19, 2026  
**Implementation:** `backend/config.py`

---

### D-16: Demo User Seeding
**Decision:** Provide seed script with two demo users  
**Demo Accounts:**
- **Applicant:** applicant@creditxplain.com / password123
- **Officer:** officer@creditxplain.com / password123

**Rationale:**
- **Testing convenience** - ready-to-use accounts
- **Role demonstration** - shows both applicant and officer workflows
- **Reproducible** - same credentials across all environments
- **Security note** - passwords clearly marked as demo-only

**Script:** `backend/seed_db.py`  
**Usage:** `python3 -m backend.seed_db`

**Date:** Nov 19, 2026

---

### D-17: Error Handling Strategy
**Decision:** HTTP status codes with structured error responses  
**Standard Codes:**
- **200** - Success (GET, PUT)
- **201** - Created (POST)
- **400** - Bad Request (validation error)
- **401** - Unauthorized (missing/invalid token)
- **403** - Forbidden (insufficient permissions)
- **404** - Not Found (resource doesn't exist)
- **500** - Internal Server Error

**Error Response Format:**
```json
{
  "detail": "Error message"
}
```

**Rationale:**
- RESTful conventions
- FastAPI automatic validation errors
- Clear error messages for frontend

**Date:** Nov 19, 2026

---

### D-18: Testing Strategy - Backend
**Approach:** Unit tests with in-memory SQLite database

**Test Categories:**
1. **Authentication tests** - Password hashing, JWT generation
2. **Database tests** - Model creation, relationships, constraints
3. **API endpoint tests** - Request/response validation (Phase 4.2+)
4. **Integration tests** - End-to-end workflows (Phase 4.9)

**Test Database:** In-memory SQLite (`sqlite:///:memory:`)  
**Benefits:** Fast, isolated, no cleanup needed

**Results (Phase 4.1):**
- Authentication: 1/3 passing (bcrypt version issue, non-blocking)
- Database: 4/4 passing ✓

**Date:** Nov 19, 2026  
**Implementation:** `backend/tests/`

---

### D-19: Dependency Compatibility Resolution
**Issues Encountered:**
1. **Pydantic + email-validator** - Required separate install
2. **bcrypt + passlib version** - Needed bcrypt 4.0.1 for passlib 1.7.4
3. **pandas version conflicts** - Used flexible version constraints

**Resolution Strategy:**
- Use `>=` instead of `==` for version constraints
- Pin only incompatible packages (bcrypt)
- Test with `pip install -r requirements.txt`

**Date:** Nov 19, 2026  
**File:** `backend/requirements.txt` (20 dependencies)

---

## Change Log

| Date | Phase | Changes |
|---|---|---|
| Nov 19, 2026 | Phase 1 Start | Initial document created; D-1 through D-4, FE-1, FE-2 documented |
| Nov 19, 2026 | Phase 1 Complete | D-2 threshold strategy refined (40-40-20 → 35-45-20 actual); D-3 validated via ablation |
| Nov 19, 2026 | Phase 2 Complete | E-1 documented (SHAP primary, LIME secondary); top 5 features identified |
| Nov 19, 2026 | Phase 3 Complete | CF-1 to CF-4 (counterfactuals), F-1 to F-7 (fairness); age disparity identified (0.3488) |
| Nov 19, 2026 | Phase 4.1 Complete | D-8 to D-19 (backend foundation); FastAPI, SQLite, JWT auth; 23 files created, 4/4 tests passing |

---

**Next Update:** Phase 4.2 (Prediction Engine Integration)


---

## 14. Service Layer Architecture (Phase 4.2)

### D-20: Service Layer Pattern
**Decision:** Separate business logic into service classes, not directly in routers  
**Alternatives Considered:**
- Put ML code directly in router endpoints
- Create utility functions without classes
- Microservices architecture (separate processes)

**Rationale:**
- **Separation of concerns** - routers handle HTTP, services handle business logic
- **Reusability** - services can be used by multiple routers or CLI tools
- **Testability** - services can be tested independently of FastAPI
- **Singleton pattern** - models loaded once and reused across requests
- **State management** - services maintain loaded models in memory

**Services Created:**
1. **PredictionService** - Model loading and predictions
2. **ExplainabilityService** - SHAP/LIME generation
3. **CounterfactualService** - DiCE recommendations

**Date:** Nov 19, 2026  
**Implementation:** `backend/services/`

---

### D-21: Model File Selection
**Decision:** Use `final_lgbm.pkl` from Phase 1, not old `lightgbm.pkl`  
**Issue:** Old baseline model (Aug 16) vs Phase 1 final model (Nov 19)

**Rationale:**
- `final_lgbm.pkl` is the Phase 1 selected model with FE V2
- Old `lightgbm.pkl` is baseline model without feature engineering
- Phase 1 model has better performance (0.7722 vs 0.7203 AUC)

**Files Used:**
- Model: `models/final_lgbm.pkl` (2.3 MB)
- Scaler: `models/scaler.pkl` (13 KB)
- Features: `models/continuous_cols.pkl` (1.5 KB)

**Date:** Nov 19, 2026

---

### D-22: Graceful Model Loading
**Decision:** Services initialize without crashing if model files missing  
**Behavior:**
- Print warning message to console
- Set model/scaler/features to None or empty
- API endpoints return 500 with clear error message
- Server continues to run (other endpoints still work)

**Rationale:**
- Development convenience - backend can start without trained models
- Testing flexibility - tests can run without model files
- Deployment flexibility - backend and model training can be separate
- Clear error messages for debugging

**Date:** Nov 19, 2026

---

### D-23: SHAP as Primary Explainability in API
**Decision:** Generate SHAP for every prediction, LIME as supplementary  
**Rationale:**
- SHAP is exact for tree models (no sampling variance)
- SHAP values satisfy additivity property
- LIME provides alternative perspective for validation
- Both stored in database for comparison

**API Behavior:**
- SHAP: Always generated, top 20 features returned
- LIME: Always generated, top 10 features returned
- Both saved to `explanations` table

**Date:** Nov 19, 2026

---

### D-24: Counterfactuals Only for Rejections
**Decision:** Generate counterfactuals only when decision = REJECT  
**Alternatives:**
- Generate for all decisions (wasteful)
- Generate for REVIEW and REJECT (partial)
- Generate on-demand via separate endpoint

**Rationale:**
- APPROVE decisions don't need recourse (already approved)
- REVIEW decisions await officer action (officer may override)
- REJECT decisions need actionable recommendations
- Saves computation time (~2-3 seconds per counterfactual)

**Date:** Nov 19, 2026

---

### D-25: Feature Preprocessing in Service Layer
**Decision:** Services handle all feature preprocessing (scaling, selection)  
**Preprocessing Steps:**
1. Convert feature dict to DataFrame
2. Fill missing columns with 0
3. Select only continuous_cols (match training)
4. Apply StandardScaler transformation

**Rationale:**
- Consistent preprocessing across all predictions
- Encapsulates ML pipeline complexity
- API callers just pass raw feature values
- Reduces chance of preprocessing mismatch

**Date:** Nov 19, 2026

---

## Change Log (Updated)

| Date | Phase | Changes |
|---|---|---|
| Nov 19, 2026 | Phase 1 Start | Initial document created; D-1 through D-4, FE-1, FE-2 documented |
| Nov 19, 2026 | Phase 1 Complete | D-2 threshold strategy refined; D-3 validated via ablation |
| Nov 19, 2026 | Phase 2 Complete | E-1 documented (SHAP primary, LIME secondary); top 5 features identified |
| Nov 19, 2026 | Phase 3 Complete | CF-1 to CF-4 (counterfactuals), F-1 to F-7 (fairness); age disparity identified (0.3488) |
| Nov 19, 2026 | Phase 4.1 Complete | D-8 to D-19 (backend foundation); FastAPI, SQLite, JWT auth; 23 files created |
| Nov 19, 2026 | Phase 4.2 Complete | D-20 to D-25 (service layer); prediction, explainability, counterfactual services |

---

**Next Update:** Phase 4.3 (Loan Terms Calculator)

---

## 15. Loan Calculator Design (Phase 4.3)

### D-26: Risk-Based Interest Rate Pricing
**Decision:** Interest rate based on decision band + default probability  
**Formula:** `Base Rate (10%) + Risk Spread + Probability Adjustment`

**Risk Spreads:**
- APPROVE: +2% (12% minimum)
- REVIEW: +5% (15% minimum)
- REJECT: +8% (18% minimum)

**Probability Adjustment:**
- Within each band, rate increases linearly with probability
- APPROVE: 0-2% additional (proportional to position in 0-0.0349 band)
- REVIEW: 0-3% additional (proportional to position in 0.0349-0.1189 band)
- REJECT: Fixed +3%

**Rationale:**
- Risk-based pricing is standard banking practice
- Lower risk applicants get better rates (fairness + incentive)
- Covers expected default losses (higher risk = higher rate)
- Progressive within bands reflects granular risk assessment
- Cap at 25% prevents predatory lending

**Date:** Nov 19, 2026

---

### D-27: Interest Rate Cap at 25%
**Decision:** Maximum interest rate capped at 25% annual  
**Alternatives:**
- No cap (market rate)
- Higher cap (30-36%)
- Lower cap (20%)

**Rationale:**
- 25% is reasonable for unsecured personal loans
- Prevents excessive rates for high-risk applicants
- Regulatory considerations (usury laws vary by jurisdiction)
- Ethical lending practices (avoid debt trap)

**Date:** Nov 19, 2026

---

### D-28: EMI Calculation Formula
**Decision:** Standard EMI formula used by banks  
**Formula:** `EMI = P × r × (1 + r)^n / ((1 + r)^n - 1)`

**Where:**
- P = Principal loan amount
- r = Monthly interest rate (annual / 12)
- n = Loan tenure in months

**Rationale:**
- Industry-standard formula (not custom)
- Ensures equal monthly installments
- Mathematically derived from compound interest
- Easy to verify and audit
- Compatible with existing loan management systems

**Verification:** ₹100,000 at 12% for 12 months = ₹8,884.88/month (matches standard calculators)

**Date:** Nov 19, 2026

---

### D-29: Default Loan Tenure
**Decision:** 12 months as default tenure. **Superseded in Phase 4.8 by D-52** (tenure chosen by the applicant)  
**Alternatives:**
- 6 months (shorter, higher EMI)
- 24 months (longer, lower EMI)
- Dynamic based on loan amount

**Rationale:**
- 12 months balances affordability and interest cost
- Standard for small personal loans
- Can be customized per application if needed
- Frontend can offer tenure selection (6/12/24 months)

**Date:** Nov 19, 2026

---

### D-30: Loan Amount Source
**Decision:** Extract from application features (AMT_CREDIT field)  
**Fallback:** ~~Default to ₹100,000 if field missing~~ Removed in Phase 4.8: the amount is now required (D-53)

**Rationale:**
- AMT_CREDIT is the requested credit amount in Home Credit dataset
- Applications naturally include requested amount
- Fallback prevents errors for incomplete data
- Production system would require this field (mandatory)

**Date:** Nov 19, 2026

---

### D-31: Loan Terms Storage
**Decision:** Save to `loan_terms` table, one-to-one with `decisions`  
**Fields Stored:**
- loan_amount (principal)
- interest_rate (annual)
- tenure_months
- monthly_emi
- total_payable

**Not Stored:**
- Amortization schedule (can be calculated on-demand)
- Processing fees (not in scope)
- Prepayment terms (not in scope)

**Rationale:**
- Essential terms stored for audit trail
- Lightweight storage (5 fields)
- Amortization schedule regenerable from stored terms
- Can be extended later if needed

**Date:** Nov 19, 2026

---

## Change Log (Updated)

| Date | Phase | Changes |
|---|---|---|
| Nov 19, 2026 | Phase 4.1 | D-8 to D-19 (backend foundation) |
| Nov 19, 2026 | Phase 4.2 | D-20 to D-25 (service layer) |
| Nov 19, 2026 | Phase 4.3 | D-26 to D-31 (loan calculator) |

---

**Next Update:** Phase 4.4 or 4.5

---

## 16. Fairness Monitoring Design (Phase 4.5)

### D-32: Real-time Fairness Calculation from Database
**Decision:** Calculate fairness metrics from applications in database, not from static files  
**Alternatives:**
- Pre-calculated metrics from Phase 3 reports
- Batch processing with scheduled jobs
- Static fairness report displayed in dashboard

**Rationale:**
- Real-time metrics reflect current system behavior
- Updates as new applications are processed
- Officer can trigger recalculation on-demand
- Historical tracking via fairness_metrics table
- More accurate than static Phase 3 reports

**Date:** Nov 19, 2026

---

### D-33: Minimum 50 Applications for Fairness Calculation
**Decision:** Require at least 50 applications before calculating fairness metrics  
**Rationale:**
- Statistical significance requires sufficient sample size
- Small samples can show misleading disparities
- 50 is reasonable minimum for demographic breakdown (5 age bands × ~10 per band)
- API returns clear message if insufficient data

**Date:** Nov 19, 2026

---

### D-34: Officer-Only Access to Fairness Endpoints
**Decision:** All fairness endpoints require officer role  
**Rationale:**
- Sensitive information (demographic disparities)
- Compliance/audit function (not for applicants)
- Prevents gaming of system based on fairness metrics
- Aligns with regulatory best practices (internal monitoring)

**Date:** Nov 19, 2026

---

### D-35: Alert Threshold at 10% (0.1 DPD)
**Decision:** Alert triggers when demographic parity difference exceeds 0.1  
**Alternatives:**
- 20% threshold (80% rule: 0.2 DPD)
- 5% threshold (stricter)
- No threshold (always show metrics)

**Rationale:**
- Stricter than 80% rule (0.1 vs 0.2)
- Balances sensitivity and actionability
- Phase 3 found 0.3488 age disparity (clearly exceeds threshold)
- Aligns with D-3 decision (fairness is priority)

**Date:** Nov 19, 2026

---

### D-36: Store Fairness Metrics History
**Decision:** Save each calculation to fairness_metrics table  
**Fields:** age_dpd, children_dpd, metrics_json, has_alert, alert_message, calculated_at

**Rationale:**
- Audit trail for compliance
- Track fairness trends over time
- Compare before/after mitigation efforts
- Evidence for regulatory review
- Historical analysis for manuscript

**Date:** Nov 19, 2026

---

### D-37: Demographic Groups Match Phase 3
**Decision:** Use same age bands and family status as Phase 3  
**Age Bands:** <25, 25-35, 35-45, 45-55, 55+  
**Family Status:** Has Children / No Children

**Rationale:**
- Consistency between offline audit (Phase 3) and real-time monitoring (Phase 4)
- Age bands reflect life stages
- Family status simpler than marital status + children (easier to interpret)
- Matches Phase 3 findings (age disparity 0.3488)

**Date:** Nov 19, 2026

---

## 17. Changes Recorded After Phase 4.5 (Phase 4.6 doc sync)

The first three decisions were made in commits ae6ad19, fef60f8 and b68c3f7 (Nov 19, 2026) but were not written down at the time. They are recorded here from the commits and the code.

### D-38: Bounded Lever Search Replaces DiCE in the API
**Decision:** The API's recourse uses `src/recourse.py`, a grid search over three levers the applicant controls, instead of DiCE on engineered columns.  
**Levers and bounds:** loan amount ×{1.0 … 0.5} (8 steps), verified income ×{1.0 … 1.5} (5 steps), yearly repayment (the dataset's `AMT_ANNUITY`, which is yearly) ×{1.0 … 0.7} (4 steps, i.e. longer tenure). 160 candidates per applicant.  
**How it works:** Every candidate is rebuilt through FE v2 (`add_features`), so derived ratios stay consistent. Candidates that move the applicant to a better band are kept, ranked by cost `(1−credit) + (income−1) + (1−installment)`, and at most one is kept per lever combination (up to 3 options).  
**Rationale:**
- DiCE perturbed engineered columns independently, so ratios no longer matched their inputs (KNOWN_ISSUES #17).
- Only applicant-controllable inputs change; immutable features are never touched.
- Latency about 14 ms against 2–5 s for DiCE (#40).

**Caveat:** The suggestions are model-based, not causal. The validity measure needs replacing (#55).  
**Commit:** b68c3f7

---

### D-39: One Shared Inference Pipeline
**Decision:** The backend scores through `src/inference.py` (`Scorer`): raw applicant fields → defaults for missing fields (`models/feature_defaults.json`, built by `src/build_feature_defaults.py`) → FE v2 → final LightGBM Booster → three-band decision.  
**Rationale:** Training, offline evaluation and the API use the same feature path, so the API cannot drift from the model card.  
**Commit:** ae6ad19

---

### D-40: Registration Always Creates Applicants; Default Secret Key Refused
**Decision:** `POST /api/auth/register` ignores any client-supplied role and always creates an `APPLICANT`. Officers are created only by seeding. With `DEBUG=False`, the app refuses to start if `SECRET_KEY` is still the default.  
**Rationale:** Prevents self-registration as an officer (privilege escalation) and accidental deployment with a known signing key.  
**Tests:** `backend/tests/test_access_control.py`  
**Commit:** fef60f8

---

### D-41: Working Agreement and Phase Order From Phase 4.6
**Decision:** Each phase is proposed and approved by the project lead before any change. After a phase these docs are updated and every change is documented; the project lead runs the tests, commits and pushes to `dev-varun`.  
**Order:**

| Phase | Scope |
|---|---|
| 4.6 | Cleanup, doc sync, full test baseline |
| 4.7 | Publication-grade evaluation of LIME and recourse (D-42) |
| 4.8 | Backend gaps: document upload, officer queue TODO, tenure input |
| 4.4 | PDF report generator (number kept from the skipped slot) |
| 4.9 | Frontend: foundation and applicant portal |
| 4.10 | Frontend: officer dashboard |
| 5.x | CI, README, latency, usability check, evaluation, manuscript |

**Date:** Nov 25, 2026

---

### D-42: Evaluation Standard for Explanations and Recourse (confirmed in 4.7, see D-43 to D-49)
**Decision (proposed, to be confirmed at the start of 4.7):** The paper reports only explanation and recourse numbers that a reviewer cannot dismiss as artefacts of the setup.  
**LIME:** sample around the instance, quartile discretisation, explicit kernel width, fixed seed. Evaluate 100–200 stratified applicants with top-k overlap, rank correlation with SHAP, and stability across seeds (#52–#54).  
**Recourse:** report coverage, proximity, sparsity, plausibility (distance to real applicants) and latency. Compare against a baseline under the same bounds: DiCE (`src/counterfactuals.py`) or random search. Drop the circular validity rate (#55). Present recourse as Reject→Review and state why Approve is out of reach (#56).  
**Rationale:** The goal is a result worth presenting and publishing.  
**Date:** Nov 25, 2026

---

## 18. Explanation and Recourse Upgrade (Phase 4.7)

D-42 (the evaluation standard) is now **confirmed and implemented** by the decisions below. Goal: improve model functionality and novelty.

### D-43: Local LIME Over the Model's Top-Gain Features
**Decision:** LIME (`src/lime_local.py`, `LocalLime`) perturbs only the 25 features with the highest LightGBM gain and holds every other feature at the applicant's own value. It uses LIME's default quartile discretisation, treats binary columns as categorical, fits the surrogate to the model's **log-odds** (the scale SHAP uses), and is seeded (both of LIME's random generators are reset per call). Background: 5,000 training rows (previously 2,000 in the API; the offline script used the **test** set).  
**Why:** Measured on 150 held-out applicants (50 per band), against the old set-up (all 300 features, continuous, sampled around the training mean):

| Metric | Old | New |
|---|---|---|
| Rank agreement with SHAP (Spearman, applicant's top-25 SHAP features) | 0.00 | 0.64 |
| Locality gap (own applicant's SHAP minus another applicant's) | 0.00 | 0.49 |
| Sign agreement on SHAP top-5 | 0.52 | 0.74 |
| Top-10 overlap with SHAP | 0.36 | 0.55 |
| Stability across 5 seeds (top-5 Jaccard) | n/a | 0.83 |
| Surrogate R² | 0.54 | 0.31 |
| Latency per explanation | 57 ms | 38 ms |

The old set-up had a higher R² because it fitted one global linear model, which is why it gave every applicant nearly the same answer (locality gap 0). The new set-up's R² is lower because it explains bin membership locally, the quantity LIME is designed for.  
**Bug found on the way:** in regression mode LIME stores the real weights under label 1 and the same weights **negated** under label 0. `LocalLime` reads label 1; a test guards it (`test_signs_match_shap`).  
**Limitation (for the paper):** LIME covers 25 of 300 features. Gain importance comes from the model, not from SHAP, so the subset does not use SHAP's answer.

---

### D-44: Recourse Must Clear the Cutoff by a Safety Margin
**Decision:** A recourse option qualifies only if it reaches the better band with the cutoff tightened by 0.1 in log-odds (about 10% on the odds). Each option reports `robust_margin`, the log-odds distance below the cutoff it crosses.  
**Why:** Options that land just under the cutoff flip back with any small model update. A margin is a simple, transparent form of robust recourse.

---

### D-45: Plausibility Check on the Loan/Income Profile
**Decision:** A suggestion is kept only if its profile (log income, credit, annuity, goods price) has a mean 5-nearest-neighbour distance (standardised) to real **non-rejected training applicants** no larger than the 95th percentile of that distance among held-back real non-rejected applicants.  
**Built by:** `python3 -m src.build_recourse_reference` → `models/recourse_reference.npz` (20,000 reference profiles, 5,000 calibration, threshold 0.2346). Gitignored, because it holds individual rows. If the file is missing the check is switched off and `plausible` is `null`.  
**Why:** Without it, the search can suggest profiles no real approved applicant has, such as a very large down payment on top of a large income increase.

---

### D-46: Refinement Towards the Applicant's Own Values
**Decision:** After the grid search, each lever combination's cheapest qualifying point is pulled back towards the applicant's own values in 20 steps. The smallest step that still qualifies is kept.  
**Why:** The grid is coarse (credit steps of 5–10%), so the grid optimum often asks for more change than needed. Refinement never increases cost (`test_refinement_never_costs_more`).

---

### D-47: Two-Tier Recourse (Reject → Review, Review → Approve)
**Decision:** The API now returns recourse for **Review** applicants too (options that reach Approve), not only for rejections.  
**Why:** Recourse from Reject cannot reach Approve within the bounds (KNOWN_ISSUES #56), but Review applicants can often be moved to automatic approval with small changes. Together the two tiers form a graduated path.

---

### D-48: Each Recourse Option Says Why It Works
**Decision:** Each API recourse option carries `drivers`: the 3 model features whose SHAP contribution changes most when the option is applied (`src/recourse.explain_change`, log-odds). The deltas sum exactly to the change in the model's score (SHAP additivity; tested).  
**Why:** It links explanation and recourse. The applicant and the officer see, for example, that a lower loan amount helps through the credit-to-goods and annuity ratios.  
**Refined in Phase 4.4:** only features whose value the option actually changes are listed. SHAP can shift credit onto unchanged features (such as age) through interactions, and naming those would mislead.

---

### D-49: Recourse Evaluation Protocol
**Decision:** `python3 -m src.recourse_eval` compares methods on held-out applicants: 300 rejected (tier Reject→Review) and 300 in Review (tier Review→Approve).
- **Methods:**
  - `previous`: the pre-4.7 engine.
  - `proposed`.
  - Two ablations, each switching off one rule.
  - `random`: random search in the same box with a similar budget and the same rules.
  - `dice`: DiCE random method on the same three levers through the same FE v2 path, run on 20 applicants per tier. Not completed in 4.7 (KNOWN_ISSUES #61).
- **Run settings (Phase 4.7b):**
  - DiCE uses its binary post-hoc sparsity search (`posthoc_sparsity_algorithm="binary"`) instead of the default linear search, which made up to 10,000 single-row model calls.
  - The DiCE model wrapper is vectorised, caches repeated single-row calls and recomputes the derived ratios; DiCE gets 200 background rows.
  - The run uses 2 CPU threads (`--threads`) and low priority, and saves rows every 10 applicants and on Ctrl+C.
  - `--methods` runs a subset and merges its rows into the existing per-applicant file.
  - DiCE suggestions are scored exactly as produced. Those outside the lever space (installment not equal to amount × a tenure factor of 0.7–1.0) are not used and are counted separately, because DiCE cannot express the coupling between amount and installment.
  - Applicant selection uses its own random generator (seed 42), separate from random search (seed 7), so the same applicants are evaluated whichever methods run.
- **Metrics:**
  - coverage;
  - cost, i.e. proximity: the sum of relative lever changes;
  - levers changed (sparsity);
  - plausible and robust shares of the best option;
  - latency.

  Every method's best option is re-scored from scratch, so all methods are measured the same way.
- **Recourse burden by group:** coverage and mean cost by age band and by children, for the proposed method.
- **Dropped:** the old "validity rate" (circular, KNOWN_ISSUES #55) and `src/recourse_validity.py`.

**Date:** Nov 25, 2026

---

## 19. Backend Completion (Phase 4.8)

### D-50: Document Storage
**Decision:** Uploaded documents go to `uploads/<application id>/<random hex>.<ext>` on disk (gitignored as personal data). A `documents` table keeps the original name, type, size, uploader and time.  
**Rules:**
- Only the applicant who owns the application can upload. The owner and officers can list and download.
- PDF, PNG or JPEG only, at most 5 MB (`settings.MAX_UPLOAD_BYTES`).
- The first bytes must match the declared type, so a renamed file is refused (HTTP 415).
- `doc_type` must be one of: income_proof, id_proof, address_proof, bank_statement, employment_proof, other.
- Each upload adds an `AuditLog` entry (`document_uploaded`, `officer_id` empty).

**Why:** Generated file names prevent path tricks and collisions. Keeping files out of the database keeps SQLite small.  
**Not done:** virus scanning and deletion (KNOWN_ISSUES #67).

---

### D-51: Officer Overrides Need a Justification
**Decision:** An officer must write a justification to approve a Review or Reject case, to reject an Approve case, or to request documents. Agreeing with the model needs none. Approving or rejecting an unscored application is refused.  
**Why:** The plan requires overrides with a logged justification. Requested-documents messages are shown to the applicant (`document_requests`, read from the audit log).

---

### D-52: Loan Tenure Chosen by the Applicant
**Decision:** `tenure_months` is part of the application: one of 12, 24, 36, 48 or 60 (`settings.TENURE_OPTIONS`), default 12. It is stored in a new `applications.tenure_months` column and used for loan terms, both at prediction and on officer approval.  
**Migration:** `create_all()` does not add columns to existing tables, so `backend/database.upgrade_schema` adds missing columns at startup (currently just this one). Running it twice is harmless.  
**Limitation:** The model still uses the applicant's `AMT_ANNUITY` as given; tenure affects pricing only.

---

### D-53: Requested Loan Amount Is Required
**Decision:** `features.AMT_CREDIT` must be a positive number when an application is created (HTTP 422 otherwise). The silent ₹100,000 fallback is removed.  
**Why:** A loan priced on an invented amount is wrong without anyone noticing (KNOWN_ISSUES #46).

---

### D-54: Officer Approval Creates Loan Terms at the Model's Band
**Decision:** When an officer approves an application that has no loan terms, terms are created from its requested amount and tenure, priced at the model's band (an approved Review case pays the Review rate). Shared helper: `services/loan_calculator.loan_terms_for`.  
**Why:** Before, an approved Review case had no loan terms at all.

---

### D-55: Fairness Group Statistics Stored as Plain JSON
**Decision:** `FairnessService._group_stats` stores per-group counts as `{"approved": {"sum"|"count"|"mean": {group: value}}}` with string keys and no NaN; empty groups get 0.  
**Why:** The previous `groupby(...).agg(...).to_dict()` produced tuple keys, which cannot be stored as JSON, so `POST /api/fairness/calculate` failed once 50 applications existed. It also did not match the structure `get_demographic_statistics` reads.

**Date:** Nov 26, 2026

---

## 20. Richer Recourse (Phase 4.7c)

### D-56: Waiting as a Recourse Lever, With Age Held Fixed
**Decision:** A fourth lever, `wait`: re-apply after 3, 6 or 12 months, staying in the same job and making no new credit applications meanwhile (`src/recourse._pass_time`).

**What changes while waiting:**
- `DAYS_EMPLOYED`, `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH` and `DAYS_LAST_PHONE_CHANGE` grow by the waiting time. Job tenure does not grow for applicants who are not in employment (`IS_EMPLOYED_ANOMALY = 1`).
- `OWN_CAR_AGE` grows for car owners.
- Credit-bureau enquiries age out of their exclusive windows. After 3 or 6 months every recent enquiry (hour to quarter) sits in the year window; after 12 months there are none within a year.

**What does not change:** the applicant's age (`DAYS_BIRTH`, `AGE_YEARS`) is held fixed, although the model uses age. No option may rely on the applicant getting older, which would be unfair advice and would not reflect anything the applicant did.

**Cost:** 0.04 per month, so 12 months costs about as much as halving the loan. Each option reports `wait_months` separately.

**Spelled out:** a waiting option lists every changed field, with original and new values and a `lever` tag, so re-scoring gives the same result wherever it is done. DAYS fields also carry positive display values.

**Rejected lever, paying down existing loans:** `BUREAU_AVG_CREDIT_AMT` and `BUREAU_CREDIT_COUNT` are averages and counts over all past credits, closed ones included. Repaying a loan would not change them, so this lever cannot be modelled honestly with this data.

---

### D-57: Applicant Preferences for Recourse
**Decision:** `find_recourse` accepts `locked` levers (credit, income, tenure, wait), `max_loan_reduction` (0–0.5) and `max_wait_months` (0, 3, 6, 12).

**API:**
- Applicants can store `recourse_preferences` with an application (new `applications.recourse_preferences` JSON column, added to old databases by `upgrade_schema`). Stored preferences are used when the application is scored.
- `POST /api/predictions/{id}/recourse` returns options for new preferences, for a scored, not-approved application, without storing anything. Access: the owner or an officer.

**Why:** Advice to add income is useless to someone who cannot, and a 50% loan cut may defeat the purpose of the loan.

---

### D-58: Evaluation Keeps the Comparison Like-for-Like
**Decision:** In `src/recourse_eval.py`, `proposed` and the ablations use the three loan levers only (`locked=("wait",)`), as random search and DiCE do. A new method, `proposed_with_wait`, measures what waiting adds. Recourse burden by group is reported for both.

**Date:** Nov 26, 2026

---

## 21. PDF Reports (Phase 4.4)

### D-59: Two Reports From One Service
**Decision:** `backend/services/report_service.py` (ReportLab, A4) builds two PDFs from what is stored for an application.

**Applicant report** (`GET /api/applications/{id}/report.pdf`; owner or officer):
- the decision and estimated risk, in plain words;
- the five strongest SHAP factors, as plain-language names with "raises risk" or "lowers risk";
- recourse options, including waiting options and their "why it helps" drivers;
- loan terms, labelled as illustrative;
- officers' document requests;
- a note that a person can review any decision.

**Compliance report** (`GET /api/officer/{id}/compliance-report.pdf`; officers only). Everything above, plus:
- the model card summary and the cutoffs stored with the decision;
- the top 10 SHAP contributions and the top 10 LIME rules (log-odds);
- recourse details (risk, margin, plausibility, cost);
- recourse preferences;
- documents;
- the full audit trail;
- the latest fairness status.

**Design points:**
- **Plain language:** about 40 frequent features have hand-written names (`FEATURE_NAMES`); others fall back to a cleaned-up column name.
- **Archive:** generating a compliance report is recorded in the audit log (`compliance_report`). The PDF is not stored, because it can be rebuilt from the database and storing it would duplicate personal data.
- **Uncompressed PDFs** (a few tens of KB), so tests can check the text without a PDF reader.
- **Currency** shown as "INR", because the standard PDF fonts have no rupee glyph.
- **Wording:** the reports follow the spirit of rules that ask lenders to give written reasons for decisions (e.g. the RBI Fair Practices Code; adverse-action notices). They are not claimed to make the system legally compliant.

**Date:** Nov 26, 2026

---

### D-60: No Dominated Recourse Options
**Decision:** An option is dropped if another option asks for no more on every lever: loan amount, income, repayment and waiting time.  
**Why:** Without this, an option could repeat a cheaper option's changes and add a wait, which is never worth showing. The cheapest option is never dropped, so coverage and cost figures are unaffected.  
**Date:** Nov 26, 2026

---

## 22. Frontend: Foundation and Applicant Portal (Phase 4.9)

### D-61: Client-Rendered Pages Against the FastAPI Backend
**Decision:**
- Pages are client components that call the FastAPI backend through one typed client (`frontend/lib/api.ts`, axios, types in `lib/types.ts`).
- Sign-in state lives in `AuthProvider`; pages are guarded by role with `RequireAuth`.
- No new dependencies are added (axios, Tailwind and lucide-react were already installed).

**Next.js 16 specifics** (checked against the bundled docs in `node_modules/next/dist/docs`):
- `cacheComponents` is on, so the dynamic `/applications/[id]` page wraps its `useParams` client view in `<Suspense>`; otherwise the build fails.
- Browser-only reads (the login token) happen in effects, never during render.

**Why:** All data is per user and behind a login, so server rendering gains little. Keeping one API client means the frontend uses exactly the backend's contract.

---

### D-62: Login Token in localStorage
**Decision:** The JWT from `/api/auth/login` is stored in `localStorage` (`creditxplain_token`) and sent as a Bearer token. An invalid token is cleared when the session is restored.  
**Trade-off:** Simple and matches the existing backend. A script injected into the page could read it, which an httpOnly cookie would prevent (KNOWN_ISSUES #76).

---

### D-63: Plain-Language Labels Come From the Backend
**Decision:** The application detail and `POST /api/predictions/{id}/recourse` add a `label` to SHAP rows, LIME rows and recourse drivers (`backend/serializers.py`, using `report_service.feature_name`).  
**Why:** One source of wording for the web pages and the PDF reports, instead of copying about 70 names into the frontend. Old stored explanations get labels too, because labels are added when the data is served.

---

### D-64: Application Form Design
**Decision:**
- Applicants enter the loan amount, goods price, yearly income and yearly repayment, plus a short profile (age, years in job, children, family size, education, income type, family status, housing, car and property). A repayment estimate (12% a year over the chosen tenure) can be filled in.
- Credit-bureau scores are optional, labelled as bank-supplied in a real system; empty fields use the backend's defaults.
- Recourse limits are optional.
- Gender is not collected.
- Three sample profiles (strong, borderline, higher-risk) land in Approve, Review and Reject with the current model, for demos and the usability check.

**Why:** Only fields an applicant can know are asked for. Everything else uses typical values, as the API already does.

**Date:** Nov 26, 2026

---

## 23. Frontend: Officer Dashboard (Phase 4.10)

### D-65: Officer Workflow in the Browser
**Decision:** Four officer pages (`/officer`, `/officer/applications/[id]`, `/officer/analytics`, `/officer/fairness`), all guarded to officer and admin roles.
- **Justification enforced twice:** the action form applies the backend rule (overrides and document requests need a reason) and keeps the button disabled until one is given. The backend still checks.
- **Read-only documents:** officers can download but not upload.
- **Recourse view:** officers see the applicant's limits and can use the same "What if…?" recomputation.
- **Compliance report:** downloaded from the review page; each download is logged and the audit trail refreshes.

---

### D-66: Honest Fairness Panel
**Decision:**
- The panel shows the demographic parity difference for age bands and for children / no children, with the 0.1 alert threshold, plus approval rate per group as a chart and a table.
- The "not approved" column is labelled as including Review cases (#69).
- Equalised odds is not shown, because live applications have no repayment outcome (#68). It stays in the offline audit.

**Why:** It avoids presenting a metric that is always zero, or a count that reads as rejections.

**Date:** Nov 26, 2026

---

## 24. Reproducibility and CI (Phase 5.1)

### D-67: One Pinned Dependency File
**Decision:** `requirements.txt` at the root lists every Python dependency (pipeline, backend, tests) with exact versions: the set the 212-test suite passes with on Python 3.9. `backend/requirements.txt` includes it (`-r ../requirements.txt`). Streamlit, retired by amendment B1, is removed.  
**Why:** The two files had drifted apart. The backend list lacked `httpx`, `email-validator` and `imbalanced-learn`, and no versions were pinned.

---

### D-68: One Script Rebuilds Every Artifact
**Decision:** `scripts/run_pipeline.sh` runs the 10 main-path steps in order, from the raw Kaggle files to every report: data prep, split and baselines, FE v2, final model with ablation, feature defaults, recourse reference, explanations, explanation evaluation, fairness audit, recourse evaluation.
- `--list` shows the steps; a number resumes from that step.
- Experiments (CV, CatBoost, blend, threshold strategies, `scale_and_balance.py`) stay outside the main path and are listed in the README.

---

### D-69: CI Without Private Data
**Decision:** GitHub Actions (`.github/workflows/ci.yml`) runs two jobs on every push and pull request.
- **Model and API tests:** Python 3.9 on `ubuntu-24.04` (no 3.9 build exists for 26.04, the current `ubuntu-latest`), pinned requirements, `pytest tests backend/tests -rs`.
- **Frontend:** Node 22, `npm ci`, lint, build, then `tsc --noEmit`. The build generates the route types the type check needs.

**Data handling:** the trained model and processed data are not in git, so tests that need them skip (module-level guards, and a new `@pytest.mark.needs(...)` marker in `tests/conftest.py`). Locally, with the files present, every test runs.

**Date:** Nov 26, 2026

---

## 25. Recourse Robustness (Phase 5.2)

### D-70: Robustness Measured on Retrained Models
**Decision:** Recourse robustness is measured against 5 bootstrap-retrained models (`src/train_bootstrap_models.py`). Each repeats `src/final_model.py` exactly, except that it is fitted on a resample with replacement of the fit split, with seed 100+k:
- same features, parameters and early stopping on the same validation split;
- cutoffs re-selected by the same rule.

An option **holds** on a model if that model, with the option applied, puts the applicant in the promised band (Review or better from Reject; Approve from Review).

**Reported per method** (`src/recourse_eval.py --robustness`, logic in `src/robustness.py`):
- share of options holding on all 5 models, with each model's own cutoffs and with the final model's cutoffs;
- mean number of models held;
- share of applicants with an option that is usable and holds on all 5.

**Reference:** how often the unchanged applicant already reaches the promised band on some retrained model.

**Margin sensitivity:** proposed at margins 0, 0.1 and 0.2 (`ablation_no_margin`, `proposed`, `proposed_margin_0p2`).

**Why:** The 0.1 log-odds margin was only a design choice. Retraining is how banks actually change models, so it is the realistic test of whether advice lasts.

---

### D-71: One Cutoff Rule, Shared
**Decision:** The validation-set cutoff rule moved from inline code in `src/final_model.py` into `decision.select_cutoffs`, which `final_model.py` and the bootstrap script both call. The logic is unchanged; a test re-derives the model card's cutoffs from the saved model.

**Date:** Nov 26, 2026

---

## 26. Ensemble-Robust Recourse (Phase 5.2b)

### D-72: Options Must Survive Retrained Models
**Decision:** `find_recourse(..., ensemble=..., min_agree=...)` accepts an option only if, besides the final model's check (margin and plausibility), it reaches the better band on at least `min_agree` of the given retrained models, each with its own cutoffs. The default requires all of them. Refinement uses the same rule, so pulling an option back cannot undo it. Options report `ensemble_agreement`, e.g. "5/5".

**Why:** Phase 5.2 found that only 23.2% of Reject → Review options held on all 5 retrained models (#83). Requiring agreement during the search targets that directly, instead of only widening the margin.

---

### D-73: Separate Search and Evaluation Models
**Decision:** 10 bootstrap models.
- Models **1–5** are used by the search (`proposed_ensemble`, `proposed_ensemble_4of5`, `proposed_ensemble_with_wait`).
- Models **6–10** are used only to measure robustness, for every method.

**Why:** Measuring robustness on the same models the search used would be circular: those options hold on them by construction. Re-evaluating every method on 6–10 keeps the comparison fair. Phase 5.2's figures were measured on models 1–5, which no method used at the time; the Phase 5.2b run replaces them with figures on 6–10.

**Not changed:** the live API still uses the non-ensemble search. Switching is a separate decision after the results.

**Date:** Nov 27, 2026

---

### D-74: The Application Uses Hybrid Recourse
**Decision:** The API, both at scoring and in "What if…?", uses `find_recourse_hybrid`:
- options that also pass all 5 search models (bootstrap models 1–5) come first, marked `retrain_check_passed: true`;
- standard options fill the remaining slots for lever combinations with no robust option, marked `false`;
- if the retrained models are not present, or `ROBUST_RECOURSE` is off, it is the standard search, with `null`.

The decision page and the PDFs say "Likely to hold if the model is updated" or "May change if the model is updated"; the compliance report adds the agreement count.

**Why:** It is chosen by the project lead (Nov 27, 2026). Applicants keep the standard search's coverage, and the advice most likely to last is listed first and labelled.

**Cost:** two searches per decision instead of one, plus 5 extra models in memory.

**Date:** Nov 27, 2026

---

### D-75: Understandability Measured From the System's Text, Plus an Expert Review
**Decision:** The usability check (plan §6.4) is done in two parts instead of a questionnaire study:
- **Computed measures** (`src/understandability_eval.py`) on held-out applicants, using the app's own services and wording: readability (Flesch reading ease and Flesch–Kincaid grade), amount of information, plain-name coverage and jargon. Each displayed line counts as one sentence, because reasons and options are shown as separate lines, not as prose.
- **An expert review** by one or two evaluators (`docs/usability/expert_review.md`): Nielsen's heuristics with 0–4 severities, a task walkthrough and five explanation and recourse statements. It is reported as problems found and fixed, with individual answers, not averages.

Accessibility is checked with Lighthouse (`docs/usability/accessibility_audit.md`).

**Why:** The project has a sole developer and cannot recruit a participant group. System Usability Scale averages from one or two people carry no statistical weight, while heuristic evaluation is designed for a small number of evaluators. The computed measures are reproducible and need no participants. The paper states that no user study with applicants was run, and names it as future work (KNOWN_ISSUES #84).

**Alternatives considered:**
- A 10–15 person online form: not feasible for the project lead.
- No human element at all: still the fallback if no evaluator is available, with the computed measures reported alone.

**Date:** Nov 27, 2026

---

### D-76: Age Is an Eligibility Rule Before the Model
**Decision:** `backend/services/eligibility_service.py` runs before scoring.
- An applicant **under 18** applies with a guardian. The application is scored on the guardian's age, income, income type and employment, it is never approved automatically, and an officer can approve only after the guardian's ID proof and income proof are uploaded.
- An applicant assessed as having **no income** (zero or missing, Student or Unemployed) is never approved automatically.
- A rule only removes automatic approval (Approve → Review). It never worsens the model's band, and Review → Approve recourse is not offered when automatic approval is out of reach.
- The rules, reasons, whose details were assessed and the model's own band are stored with the decision and shown in the app and PDFs.

**Why:** This follows the project lead's description of real credit practice: age should not limit an adult who earns, and a minor needs a guardian. A minor generally cannot take on a loan alone, so the guardian is assessed as the co-applicant. The training data covers ages 20.5–69 only, so the model cannot score a minor's own record meaningfully. Pensioners keep automatic approval, because a pension is an income.

**Limits:** These are illustrative policy settings (`MIN_ADULT_AGE`, `NO_INCOME_TYPES`), not a legal review. No age-at-loan-end limit was added (KNOWN_ISSUES #85).

**Date:** Nov 27, 2026

---

### D-77: Age Stays in the Model; the Age-Free Model Was Tested and Rejected
**Finding** (`src/age_fairness_experiment.py`, 61,503 test applicants):
- Dropping the three direct age features raises the age-band DPD from 0.349 to 0.364. Also dropping the three age-derived features raises it to 0.377, and AUC falls 0.7668 → 0.7646.
- Age is recoverable from the remaining features (R² 0.735).
- Without age, the model over-predicts risk for under-25s (13.8% against an actual 11.4%) and approves fewer of them (13.4% → 9.8%).
- The current model is calibrated within 0.8 points in every age band. The approval gap follows the actual default rates (11.4% for under-25s, 5.3% for over-55s).
- Per-band cutoffs reach DPD 0.022, but only by using age at decision time, and approved under-25s then default at 3.4% instead of 0.9%.

**Decision:** keep the current model (age audited, gender excluded). Deliver the age policy through D-76, and report the experiment in the paper as evidence that fairness through unawareness fails here. The switch to the age-free model (5.5b) is not done. Confirmed by the project lead (Nov 27, 2026, Q5).

**Date:** Nov 27, 2026

---

## Change Log (Updated)

| Date | Phase | Changes |
|---|---|---|
| Nov 19, 2026 | Phase 4.1 | D-8 to D-19 (backend foundation) |
| Nov 19, 2026 | Phase 4.2 | D-20 to D-25 (service layer) |
| Nov 19, 2026 | Phase 4.3 | D-26 to D-31 (loan calculator) |
| Nov 19, 2026 | Phase 4.5 | D-32 to D-37 (fairness monitoring) |
| Nov 25, 2026 | Phase 4.6 | D-38 to D-40 (recorded from commits b68c3f7, ae6ad19, fef60f8); D-41 working agreement and phase order; D-42 proposed evaluation standard |
| Nov 25, 2026 | Phase 4.7 | D-42 confirmed; D-43 local LIME; D-44 to D-48 recourse upgrade (margin, plausibility, refinement, two tiers, drivers); D-49 recourse evaluation protocol |
| Nov 26, 2026 | Phase 4.8 | D-50 document storage; D-51 override justification; D-52 applicant-chosen tenure and schema upgrade; D-53 loan amount required; D-54 loan terms on officer approval; D-55 fairness stats format. D-29 and D-30 marked superseded |
| Nov 26, 2026 | Phase 4.7c | D-56 waiting lever with age held fixed (debt-paydown lever rejected); D-57 applicant preferences and recourse endpoint; D-58 like-for-like evaluation |
| Nov 26, 2026 | Phase 4.4 | D-59 applicant and compliance PDF reports; D-60 no dominated options; D-48 drivers limited to changed features |
| Nov 26, 2026 | Phase 4.9 | D-61 client-rendered frontend; D-62 token storage; D-63 labels from backend; D-64 application form |
| Nov 26, 2026 | Phase 4.10 | D-65 officer workflow in the browser; D-66 fairness panel wording |
| Nov 26, 2026 | Phase 5.1 | D-67 pinned requirements; D-68 pipeline script; D-69 CI without private data |
| Nov 26, 2026 | Phase 5.2 | D-70 robustness on retrained models; D-71 shared cutoff rule |
| Nov 27, 2026 | Phase 5.2b | D-72 ensemble-robust search; D-73 separate search and evaluation models |
| Nov 27, 2026 | Phase 5.2c | D-74 hybrid recourse in the application |
| Nov 27, 2026 | Phase 5.4 | D-75 understandability measured from the system's text, plus an expert review |
| Nov 27, 2026 | Phase 5.5a | D-76 age as an eligibility rule (guardian for minors, no automatic approval without income); D-77 age stays in the model; age-free model tested and rejected (confirmed, Q5) |

---

**Next Update:** Phase 5.7
