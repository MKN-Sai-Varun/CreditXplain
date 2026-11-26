# CreditXplain Backend API

FastAPI backend for the CreditXplain credit risk prediction system.

## Features

- **Authentication**: JWT-based user authentication (applicant/officer roles)
- **Risk Prediction**: LightGBM model with SHAP/LIME explanations
- **Decision Router**: Three-band routing (Approve/Review/Reject)
- **Counterfactuals**: DiCE-based actionable recommendations
- **Fairness Monitoring**: Real-time demographic parity tracking
- **Officer Dashboard**: Review queue, manual override, audit logging
- **Loan Calculator**: Risk-based interest rate and EMI calculation

## Installation

### 1. Install Dependencies

From the backend directory:

```bash
cd backend
pip3 install -r requirements.txt
```

### 2. Set Up Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and update the `SECRET_KEY` for production use.

### 3. Initialize Database

The database will be automatically created on first run. To manually initialize:

```python
from backend.database import init_db
init_db()
```

## Running the Server

### Development Mode

```bash
# From the backend directory
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Or from the project root:

```bash
cd /path/to/CreditXplain
python3 -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Production Mode

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login and get JWT token
- `GET /api/auth/me` - Get current user info

### Applications
- `POST /api/applications/` - Submit new application. Body: `{"features": {...}, "tenure_months": 12, "recourse_preferences": {...}}` (preferences optional); `features.AMT_CREDIT` (requested loan amount) is required and must be positive; `tenure_months` is one of 12, 24, 36, 48, 60 (default 12)
- `GET /api/applications/` - Get user's applications
- `GET /api/applications/{id}` - Get application details (decision, explanation, loan terms, documents, document requests)

### Documents
- `POST /api/applications/{id}/documents` - Upload a document (multipart: `file`, `doc_type`). Applicant who owns the application only. PDF, PNG or JPEG, up to 5 MB; the file content must match its type
- `GET /api/applications/{id}/documents` - Documents uploaded and documents requested by officers (owner or officer)
- `GET /api/applications/{id}/documents/{document_id}` - Download a document (owner or officer)

Files are stored under `uploads/<application id>/` (gitignored).

### Predictions
- `POST /api/predictions/{application_id}` - Generate prediction & explanation; recourse for Reject and Review cases uses the application's stored `recourse_preferences`
- `POST /api/predictions/{application_id}/recourse` - Recourse options for new preferences (`locked_levers`, `max_loan_reduction`, `max_wait_months`) for a scored, not-approved application; nothing is stored

### Officer Dashboard
- `GET /api/officer/queue` - Get review queue, with decision, explanation and recourse for each application
- `POST /api/officer/{id}/action` - Approve/reject/request documents. A justification is required to go against the model or to request documents; approving creates loan terms priced at the model's risk band
- `GET /api/officer/analytics/approval-rates` - Get approval statistics
- `GET /api/officer/{id}/audit-log` - Get application audit trail

### Fairness Monitoring
- `GET /api/fairness/metrics` - Get latest fairness metrics
- `GET /api/fairness/demographics/age` - Age band statistics
- `GET /api/fairness/demographics/family` - Family status statistics
- `POST /api/fairness/calculate` - Trigger metrics recalculation

## Testing

Run the test suite:

```bash
# From project root
python3 -m pytest backend/tests/ -v

# With coverage
python3 -m pytest backend/tests/ -v --cov=backend --cov-report=html
```

## Database Schema

### Users
- `id`, `email`, `password_hash`, `role`, `full_name`, `created_at`

### Applications
- `id`, `user_id`, `features_json`, `tenure_months`, `recourse_preferences`, `status`, `created_at`, `updated_at`

### Decisions
- `id`, `application_id`, `decision`, `probability`, `thresholds`, `created_at`

### Explanations
- `id`, `decision_id`, `shap_values`, `lime_explanation`, `counterfactuals`

### Loan Terms
- `id`, `decision_id`, `loan_amount`, `interest_rate`, `tenure_months`, `monthly_emi`

### Audit Logs
- `id`, `application_id`, `officer_id`, `action`, `justification`, `timestamp`

### Documents
- `id`, `application_id`, `uploaded_by`, `doc_type`, `original_name`, `stored_name`, `content_type`, `size_bytes`, `uploaded_at`

Columns added after the first release (currently `applications.tenure_months` and `applications.recourse_preferences`) are added to an existing database at startup (`backend/database.py`, `upgrade_schema`).

### Fairness Metrics
- `id`, `age_dpd`, `children_dpd`, `metrics_json`, `has_alert`, `calculated_at`

## Project Structure

```
backend/
├── main.py              # FastAPI application entry point
├── config.py            # Configuration and settings
├── database.py          # Database setup and session management
├── models.py            # SQLAlchemy database models
├── schemas.py           # Pydantic request/response schemas
├── auth.py              # Authentication utilities
├── routers/             # API route handlers
│   ├── auth.py
│   ├── applications.py
│   ├── predictions.py
│   ├── officer.py
│   └── fairness.py
├── services/            # Business logic (Phase 4.2+)
│   ├── prediction_service.py
│   ├── explainability_service.py
│   ├── counterfactual_service.py
│   ├── loan_calculator.py
│   └── pdf_generator.py
└── tests/               # Test suite
    ├── test_auth.py
    └── test_database.py
```

## Phase 4 Implementation Status

- ✅ **Phase 4.1**: Backend foundation, database, authentication (COMPLETE)
- ⏳ **Phase 4.2**: Prediction engine integration (NEXT)
- ⏳ **Phase 4.3**: Loan terms calculator
- ⏳ **Phase 4.4**: Application management
- ⏳ **Phase 4.5**: Fairness monitoring
- ⏳ **Phase 4.6-4.8**: Frontend development
- ⏳ **Phase 4.9**: Integration & testing
- ⏳ **Phase 4.10**: Documentation

## Development Notes

- SQLite database file (`creditxplain.db`) is created in the backend directory
- Model files must be present in `../models/` directory
- Processed data must be in `../data/processed/` directory
- All passwords are hashed using bcrypt
- JWT tokens expire after 24 hours (configurable)
- CORS is enabled for localhost:3000 (Next.js frontend)
