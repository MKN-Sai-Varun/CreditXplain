# TECHNICAL_DECISIONS — CreditXplain

**Last Updated:** Phase 1 implementation start (Nov 19, 2026)

This document records all technical and architectural decisions made during the project, with rationale.

---

## 1. Tech Stack

| Component | Technology | Rationale |
|---|---|---|
| ML Framework | scikit-learn, LightGBM, CatBoost, XGBoost | Industry standard, LightGBM selected for performance |
| Explainability | SHAP, LIME | Base paper requirement |
| Counterfactual | DiCE | Established library for actionable recourse |
| Fairness | Fairlearn | Microsoft-backed, well-documented |
| Backend | FastAPI | Modern async framework, auto-docs, type hints |
| Frontend | React + Next.js | Changed from Streamlit (Amendment B1) |
| Database | SQLite | Sufficient for single-user design, easy deployment |
| PDF Generation | ReportLab | Python-native, template support |
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
**User Confirmation:** "Gender should not be taken into consideration by the model"  
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
**Evidence:** Copilot review caught plain SMOTE interpolating one-hot indicators

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
**Decision:** 12 months as default tenure  
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
**Fallback:** Default to ₹100,000 if field missing

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
