# Backend Services - Phase 4.2

Business logic services for ML prediction pipeline.

## Services

### PredictionService (`prediction_service.py`)
**Purpose:** Load LightGBM model and make predictions

**Features:**
- Singleton pattern for efficient model loading
- Loads model, scaler, and feature list from pickle files
- Preprocesses features (scaling, feature selection)
- Makes predictions with three-band decision routing
- Graceful handling of missing model files

**Usage:**
```python
from backend.services.prediction_service import get_prediction_service

service = get_prediction_service()
probability, decision = service.predict(features_dict)
# Returns: (0.15, "REVIEW")
```

---

### ExplainabilityService (`explainability_service.py`)
**Purpose:** Generate SHAP and LIME explanations

**Features:**
- SHAP TreeExplainer for exact tree-based explanations
- LIME TabularExplainer for model-agnostic explanations
- Returns top 20 SHAP features sorted by importance
- Returns top 10 LIME features with weights

**Usage:**
```python
from backend.services.explainability_service import get_explainability_service

service = get_explainability_service()
shap_values = service.generate_shap_explanation(features_dict)
lime_explanation = service.generate_lime_explanation(features_dict)
```

---

### CounterfactualService (`counterfactual_service.py`)
**Purpose:** Generate actionable counterfactual recommendations

**Features:**
- DiCE integration for counterfactual generation
- 27 actionable features (income, employment, documents, etc.)
- Feasibility scoring (fewer changes = more feasible)
- Only generates for REJECT decisions

**Usage:**
```python
from backend.services.counterfactual_service import get_counterfactual_service

service = get_counterfactual_service()
counterfactuals = service.generate_counterfactuals(features_dict, num_counterfactuals=3)
```

---

## Service Initialization

Services are singletons - initialized once and reused across requests.

**Initialization happens at:**
1. First API request that uses the service
2. Backend startup (via router imports)

**Model Files Required:**
- `models/final_lgbm.pkl` - Trained LightGBM model
- `models/scaler.pkl` - StandardScaler for features
- `models/continuous_cols.pkl` - List of continuous feature names

**If model files are missing:**
- Services initialize gracefully with None
- API endpoints return 500 error with clear message
- No server crash

---

## Integration with API

Services are used in `/api/predictions/{application_id}` endpoint:

```python
# 1. Load application features
application = db.query(Application).filter_by(id=application_id).first()

# 2. Make prediction
prediction_service = get_prediction_service()
probability, decision = prediction_service.predict(application.features_json)

# 3. Generate explanations
explainability_service = get_explainability_service()
shap_values = explainability_service.generate_shap_explanation(application.features_json)
lime_explanation = explainability_service.generate_lime_explanation(application.features_json)

# 4. Generate counterfactuals (if rejected)
if decision == "REJECT":
    counterfactual_service = get_counterfactual_service()
    counterfactuals = counterfactual_service.generate_counterfactuals(application.features_json)

# 5. Save to database
decision_record = Decision(...)
explanation_record = Explanation(...)
db.add_all([decision_record, explanation_record])
db.commit()
```

---

## Testing

Run service tests:
```bash
python3 -m pytest backend/tests/test_services.py -v
```

**Test Coverage:**
- Service initialization
- Model loading (if files exist)
- Feature name retrieval
- Prediction (if model loaded)

**Note:** Tests skip prediction tests if model files are not available.

---

## Phase 4.2 Status

✅ **Complete:**
- PredictionService with model loading
- ExplainabilityService with SHAP/LIME
- CounterfactualService with DiCE
- Integration with predictions router
- Service tests (2 passed, 2 skipped due to missing model)

⏳ **Pending:**
- Train and save final_lgbm.pkl from Phase 1 (deferred)
- Full end-to-end testing with real model
- Loan terms calculation (Phase 4.3)
