import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

from catboost import CatBoostClassifier


TRAIN_PATH = "data/processed/X_train_fe_v2.csv"
TEST_PATH = "data/processed/X_test_fe_v2.csv"
TARGET_PATH = "data/processed/y_train.csv"

BASELINE_AUC = 0.771558

os.makedirs("reports", exist_ok=True)

print("=" * 80)
print("CATBOOST FEATURE ENGINEERING V2 - 5 FOLD CV")
print("=" * 80)

X = pd.read_csv(TRAIN_PATH)
X_test = pd.read_csv(TEST_PATH)
y = pd.read_csv(TARGET_PATH).squeeze()

print("\nDataset")
print("X_train :", X.shape)
print("X_test  :", X_test.shape)
print("Positive rate :", y.mean())

# Drop highly redundant raw age representation
if "DAYS_BIRTH" in X.columns:
    X = X.drop(columns=["DAYS_BIRTH"])
    X_test = X_test.drop(columns=["DAYS_BIRTH"])

# Make sure train/test columns are identical
X_test = X_test[X.columns]

print("Features used :", X.shape[1])

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_pred = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

fold_results = []
feature_importance_list = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):

    print("\n" + "=" * 80)
    print(f"FOLD {fold}/5")
    print("=" * 80)

    X_train_fold = X.iloc[train_idx]
    X_valid_fold = X.iloc[valid_idx]

    y_train_fold = y.iloc[train_idx]
    y_valid_fold = y.iloc[valid_idx]

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",

        iterations=3000,
        learning_rate=0.03,

        depth=7,

        l2_leaf_reg=5,
        random_strength=0.5,

        bootstrap_type="Bernoulli",
        subsample=0.8,

        random_seed=42 + fold,

        thread_count=-1,

        verbose=100,
        allow_writing_files=False
    )

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=(X_valid_fold, y_valid_fold),
        use_best_model=True,
        early_stopping_rounds=150
    )

    valid_pred = model.predict_proba(X_valid_fold)[:, 1]
    fold_test_pred = model.predict_proba(X_test)[:, 1]

    oof_pred[valid_idx] = valid_pred
    test_pred += fold_test_pred / skf.n_splits

    fold_auc = roc_auc_score(y_valid_fold, valid_pred)
    fold_pr_auc = average_precision_score(y_valid_fold, valid_pred)
    fold_logloss = log_loss(y_valid_fold, valid_pred)

    best_iteration = model.get_best_iteration()

    print("\nFold Results")
    print("Best iteration :", best_iteration)
    print("ROC-AUC        :", round(fold_auc, 6))
    print("PR-AUC         :", round(fold_pr_auc, 6))
    print("LogLoss        :", round(fold_logloss, 6))

    fold_results.append({
        "fold": fold,
        "roc_auc": fold_auc,
        "pr_auc": fold_pr_auc,
        "logloss": fold_logloss,
        "best_iteration": best_iteration
    })

    importance = model.get_feature_importance()

    fold_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": importance
    })

    feature_importance_list.append(fold_importance)


# ============================================================
# FINAL RESULTS
# ============================================================

fold_results_df = pd.DataFrame(fold_results)

mean_auc = fold_results_df["roc_auc"].mean()
std_auc = fold_results_df["roc_auc"].std()
mean_pr_auc = fold_results_df["pr_auc"].mean()

oof_auc = roc_auc_score(y, oof_pred)
oof_pr_auc = average_precision_score(y, oof_pred)
oof_logloss = log_loss(y, oof_pred)

print("\n" + "=" * 80)
print("FINAL CATBOOST V2 RESULTS")
print("=" * 80)

print(f"Mean Fold ROC-AUC : {mean_auc:.6f}")
print(f"Std Fold ROC-AUC  : {std_auc:.6f}")
print(f"Mean Fold PR-AUC  : {mean_pr_auc:.6f}")
print(f"OOF ROC-AUC       : {oof_auc:.6f}")
print(f"OOF PR-AUC        : {oof_pr_auc:.6f}")
print(f"OOF LogLoss       : {oof_logloss:.6f}")


# ============================================================
# BASELINE COMPARISON
# ============================================================

difference = oof_auc - BASELINE_AUC

print("\n" + "=" * 80)
print("BASELINE COMPARISON")
print("=" * 80)

print(f"LightGBM V2 OOF AUC : {BASELINE_AUC:.6f}")
print(f"CatBoost V2 OOF AUC : {oof_auc:.6f}")
print(f"Difference          : {difference:+.6f}")

if difference > 0:
    print("\nCATBOOST IMPROVES OVER LIGHTGBM V2")
else:
    print("\nCATBOOST DOES NOT BEAT LIGHTGBM V2")
    print("This is still potentially useful for ensembling.")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.concat(
    feature_importance_list
).groupby("feature")["importance"].mean().sort_values(ascending=False)

importance_df = importance_df.reset_index()

importance_df["importance_pct"] = (
    importance_df["importance"]
    / importance_df["importance"].sum()
    * 100
)

print("\n" + "=" * 80)
print("TOP 40 FEATURES")
print("=" * 80)

print(
    importance_df.head(40).to_string(index=False)
)


# ============================================================
# SAVE OOF / TEST
# ============================================================

oof_df = pd.DataFrame({
    "prediction": oof_pred,
    "target": y
})

test_df = pd.DataFrame({
    "prediction": test_pred
})

oof_df.to_csv(
    "reports/catboost_fe_v2_oof.csv",
    index=False
)

test_df.to_csv(
    "reports/catboost_fe_v2_test.csv",
    index=False
)

importance_df.to_csv(
    "reports/catboost_fe_v2_feature_importance.csv",
    index=False
)

fold_results_df.to_csv(
    "reports/catboost_fe_v2_results.csv",
    index=False
)

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print("reports/catboost_fe_v2_oof.csv")
print("reports/catboost_fe_v2_test.csv")
print("reports/catboost_fe_v2_feature_importance.csv")
print("reports/catboost_fe_v2_results.csv")

print("\nDONE")