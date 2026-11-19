"""
LightGBM parameters for the final production model.
Extracted from lgbm_fe_v2_cv.py (OOF AUC: 0.7716, Test AUC: 0.7722).
"""

LGBM_PARAMS = {
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
    
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": 42
}

EARLY_STOPPING_ROUNDS = 150

# Features to exclude from the model per ethical policy
EXCLUDED_FEATURES = [
    "DAYS_BIRTH",  # Dropped in favor of AGE_YEARS
]

# Gender features to exclude (these may appear after one-hot encoding)
GENDER_PATTERNS = ["CODE_GENDER_M", "CODE_GENDER_F", "CODE_GENDER"]
