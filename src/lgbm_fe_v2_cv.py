import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss


TRAIN_X = "data/processed/X_train_fe_v2.csv"
TEST_X = "data/processed/X_test_fe_v2.csv"
TRAIN_Y = "data/processed/y_train.csv"

OUTPUT_DIR = "reports"

N_SPLITS = 5
RANDOM_STATE = 42


print("=" * 80)
print("LIGHTGBM FEATURE ENGINEERING V2 - 5 FOLD CV")
print("=" * 80)


# ============================================================
# LOAD
# ============================================================

X = pd.read_csv(TRAIN_X)
X_test = pd.read_csv(TEST_X)
y = pd.read_csv(TRAIN_Y).squeeze()

print("\nDataset")
print(f"X_train : {X.shape}")
print(f"X_test  : {X_test.shape}")
print(f"Positive rate : {y.mean():.6f}")


# ============================================================
# REMOVE REDUNDANT ORIGINAL FEATURE
# ============================================================

if "DAYS_BIRTH" in X.columns:
    X = X.drop(columns=["DAYS_BIRTH"])

if "DAYS_BIRTH" in X_test.columns:
    X_test = X_test.drop(columns=["DAYS_BIRTH"])


# ============================================================
# ALIGN
# ============================================================

X_test = X_test.reindex(columns=X.columns)


# ============================================================
# PARAMETERS
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
# CV
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": 0.0
}).set_index("feature")

fold_results = []


# ============================================================
# TRAIN
# ============================================================

for fold, (train_idx, valid_idx) in enumerate(
    skf.split(X, y), 1
):

    print("\n" + "=" * 80)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("=" * 80)

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_valid = X.iloc[valid_idx]
    y_valid = y.iloc[valid_idx]

    model = lgb.LGBMClassifier(
        n_estimators=3000,
        **params
    )

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (X_valid, y_valid)
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
    # VALIDATION
    # --------------------------------------------------------

    valid_pred = model.predict_proba(
        X_valid
    )[:, 1]

    auc = roc_auc_score(
        y_valid,
        valid_pred
    )

    pr_auc = average_precision_score(
        y_valid,
        valid_pred
    )

    loss = log_loss(
        y_valid,
        valid_pred
    )

    print(f"Best iteration : {model.best_iteration_}")
    print(f"ROC-AUC        : {auc:.6f}")
    print(f"PR-AUC         : {pr_auc:.6f}")
    print(f"LogLoss        : {loss:.6f}")

    fold_results.append({
        "fold": fold,
        "best_iteration": model.best_iteration_,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "logloss": loss
    })

    # --------------------------------------------------------
    # OOF
    # --------------------------------------------------------

    oof[valid_idx] = valid_pred

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_pred += (
        model.predict_proba(X_test)[:, 1]
        / N_SPLITS
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance = pd.Series(
        model.booster_.feature_importance(
            importance_type="gain"
        ),
        index=X.columns
    )

    feature_importance["importance"] += (
        importance / N_SPLITS
    )


# ============================================================
# FINAL METRICS
# ============================================================

overall_auc = roc_auc_score(
    y,
    oof
)

overall_pr_auc = average_precision_score(
    y,
    oof
)

overall_logloss = log_loss(
    y,
    oof
)

fold_df = pd.DataFrame(fold_results)


print("\n" + "=" * 80)
print("FINAL V2 RESULTS")
print("=" * 80)

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
# COMPARE WITH BASELINE
# ============================================================

BASELINE_AUC = 0.765867

delta = overall_auc - BASELINE_AUC

print("\n" + "=" * 80)
print("BASELINE COMPARISON")
print("=" * 80)

print(f"Baseline OOF AUC : {BASELINE_AUC:.6f}")
print(f"V2 OOF AUC       : {overall_auc:.6f}")
print(f"Difference       : {delta:+.6f}")

if delta > 0:
    print("\nV2 IMPROVES OVER BASELINE")
else:
    print("\nV2 DOES NOT IMPROVE OVER BASELINE")


# ============================================================
# TOP FEATURES
# ============================================================

feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
)

feature_importance["importance_pct"] = (
    feature_importance["importance"]
    / feature_importance["importance"].sum()
    * 100
)

print("\n" + "=" * 80)
print("TOP 40 FEATURES")
print("=" * 80)

print(
    feature_importance
    .head(40)
    .to_string()
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

pd.DataFrame({
    "target": y.values,
    "oof_prediction": oof
}).to_csv(
    f"{OUTPUT_DIR}/lgbm_fe_v2_oof.csv",
    index=False
)

pd.DataFrame({
    "prediction": test_pred
}).to_csv(
    f"{OUTPUT_DIR}/lgbm_fe_v2_test.csv",
    index=False
)

feature_importance.to_csv(
    f"{OUTPUT_DIR}/lgbm_fe_v2_feature_importance.csv"
)

fold_df.to_csv(
    f"{OUTPUT_DIR}/lgbm_fe_v2_results.csv",
    index=False
)


print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print("reports/lgbm_fe_v2_oof.csv")
print("reports/lgbm_fe_v2_test.csv")
print("reports/lgbm_fe_v2_feature_importance.csv")
print("reports/lgbm_fe_v2_results.csv")

print("\nDONE")