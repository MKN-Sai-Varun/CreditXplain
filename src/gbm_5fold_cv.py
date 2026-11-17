import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss


# ============================================================
# CONFIG
# ============================================================

TRAIN_X = "data/processed/X_train.csv"
TRAIN_Y = "data/processed/y_train.csv"
TEST_X = "data/processed/X_test.csv"

OUTPUT_DIR = "reports"
MODEL_DIR = "models"

N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("5-FOLD LIGHTGBM OOF BASELINE")
print("=" * 80)

X = pd.read_csv(TRAIN_X)
y = pd.read_csv(TRAIN_Y)

X_test = pd.read_csv(TEST_X)

if y.shape[1] == 1:
    y = y.iloc[:, 0]
else:
    y = y.squeeze()

print("\nOriginal shapes")
print(f"X_train : {X.shape}")
print(f"y_train : {y.shape}")
print(f"X_test  : {X_test.shape}")

print(f"\nPositive rate: {y.mean():.6f} ({y.mean() * 100:.4f}%)")


# ============================================================
# BASIC CLEANUP
# ============================================================

if "DAYS_BIRTH" in X.columns:
    X = X.drop(columns=["DAYS_BIRTH"])

if "DAYS_BIRTH" in X_test.columns:
    X_test = X_test.drop(columns=["DAYS_BIRTH"])

print("\nAfter preprocessing")
print(f"X_train : {X.shape}")
print(f"X_test  : {X_test.shape}")


# ============================================================
# LIGHTGBM PARAMETERS
# ============================================================

params = {
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
    "random_state": RANDOM_STATE
}


# ============================================================
# CROSS VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(X_test))

feature_importance_gain = pd.DataFrame(
    index=X.columns
)

feature_importance_gain["importance"] = 0.0

fold_results = []


# ============================================================
# TRAIN FOLDS
# ============================================================

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):

    print("\n" + "=" * 80)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("=" * 80)

    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]

    X_valid_fold = X.iloc[valid_idx]
    y_valid_fold = y.iloc[valid_idx]

    print(f"Training rows   : {len(train_idx):,}")
    print(f"Validation rows : {len(valid_idx):,}")

    model = lgb.LGBMClassifier(
        n_estimators=3000,
        **params
    )

    model.fit(
        X_train_fold,
        y_train_fold,

        eval_set=[
            (X_valid_fold, y_valid_fold)
        ],

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=150,
                verbose=False
            ),
            lgb.log_evaluation(
                period=100
            )
        ]
    )

    # --------------------------------------------------------
    # VALIDATION PREDICTIONS
    # --------------------------------------------------------

    valid_pred = model.predict_proba(
        X_valid_fold
    )[:, 1]

    fold_auc = roc_auc_score(
        y_valid_fold,
        valid_pred
    )

    fold_pr_auc = average_precision_score(
        y_valid_fold,
        valid_pred
    )

    fold_logloss = log_loss(
        y_valid_fold,
        valid_pred
    )

    print(f"\nBest iteration : {model.best_iteration_}")
    print(f"ROC-AUC        : {fold_auc:.6f}")
    print(f"PR-AUC         : {fold_pr_auc:.6f}")
    print(f"LogLoss        : {fold_logloss:.6f}")

    fold_results.append({
        "fold": fold,
        "best_iteration": model.best_iteration_,
        "roc_auc": fold_auc,
        "pr_auc": fold_pr_auc,
        "logloss": fold_logloss
    })

    # --------------------------------------------------------
    # OOF PREDICTIONS
    # --------------------------------------------------------

    oof_predictions[valid_idx] = valid_pred

    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------

    test_pred = model.predict_proba(
        X_test
    )[:, 1]

    test_predictions += test_pred / N_SPLITS

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    fold_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.booster_.feature_importance(
            importance_type="gain"
        )
    })

    fold_importance = fold_importance.set_index("feature")

    feature_importance_gain["importance"] += (
        fold_importance["importance"] / N_SPLITS
    )


# ============================================================
# OVERALL OOF METRICS
# ============================================================

overall_auc = roc_auc_score(
    y,
    oof_predictions
)

overall_pr_auc = average_precision_score(
    y,
    oof_predictions
)

overall_logloss = log_loss(
    y,
    oof_predictions
)


print("\n" + "=" * 80)
print("FINAL 5-FOLD OOF RESULTS")
print("=" * 80)

for result in fold_results:

    print(
        f"Fold {result['fold']} | "
        f"AUC: {result['roc_auc']:.6f} | "
        f"PR-AUC: {result['pr_auc']:.6f} | "
        f"LogLoss: {result['logloss']:.6f} | "
        f"Best Iter: {result['best_iteration']}"
    )

fold_df = pd.DataFrame(fold_results)

print("\n" + "-" * 80)

print(
    f"Mean Fold ROC-AUC : "
    f"{fold_df['roc_auc'].mean():.6f}"
)

print(
    f"Std Fold ROC-AUC  : "
    f"{fold_df['roc_auc'].std():.6f}"
)

print(
    f"Mean Fold PR-AUC  : "
    f"{fold_df['pr_auc'].mean():.6f}"
)

print(
    f"OOF ROC-AUC       : "
    f"{overall_auc:.6f}"
)

print(
    f"OOF PR-AUC        : "
    f"{overall_pr_auc:.6f}"
)

print(
    f"OOF LogLoss       : "
    f"{overall_logloss:.6f}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance_gain = (
    feature_importance_gain
    .sort_values(
        "importance",
        ascending=False
    )
)

feature_importance_gain["importance_pct"] = (
    feature_importance_gain["importance"]
    / feature_importance_gain["importance"].sum()
    * 100
)

print("\n" + "=" * 80)
print("TOP 30 FEATURES")
print("=" * 80)

print(
    feature_importance_gain
    .head(30)
    .to_string()
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# OOF predictions
oof_df = pd.DataFrame({
    "target": y.values,
    "oof_prediction": oof_predictions
})

oof_path = os.path.join(
    OUTPUT_DIR,
    "lgbm_oof_predictions.csv"
)

oof_df.to_csv(
    oof_path,
    index=False
)


# Test predictions
test_pred_df = pd.DataFrame({
    "prediction": test_predictions
})

test_path = os.path.join(
    OUTPUT_DIR,
    "lgbm_test_predictions.csv"
)

test_pred_df.to_csv(
    test_path,
    index=False
)


# Feature importance
importance_path = os.path.join(
    OUTPUT_DIR,
    "lgbm_cv_feature_importance.csv"
)

feature_importance_gain.to_csv(
    importance_path
)


# Fold metrics
fold_path = os.path.join(
    OUTPUT_DIR,
    "lgbm_cv_results.csv"
)

fold_df.to_csv(
    fold_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(oof_path)
print(test_path)
print(importance_path)
print(fold_path)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)