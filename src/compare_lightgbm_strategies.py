import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

TRAIN_X_PATH = "data/processed/X_train.csv"
TRAIN_Y_PATH = "data/processed/y_train.csv"

TEST_X_PATH = "data/processed/X_test.csv"
TEST_Y_PATH = "data/processed/y_test.csv"

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("LIGHTGBM STRATEGY COMPARISON")
print("=" * 80)

X = pd.read_csv(TRAIN_X_PATH)
y = pd.read_csv(TRAIN_Y_PATH).values.ravel()

X_test = pd.read_csv(TEST_X_PATH)
y_test = pd.read_csv(TEST_Y_PATH).values.ravel()


# ============================================================
# BASIC CLEANING
# ============================================================

if "DAYS_BIRTH" in X.columns:
    X = X.drop(columns=["DAYS_BIRTH"])

if "DAYS_BIRTH" in X_test.columns:
    X_test = X_test.drop(columns=["DAYS_BIRTH"])


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)


print("\nDataset")
print("-" * 80)

print(f"Full training data : {X.shape}")
print(f"Training split     : {X_train.shape}")
print(f"Validation split   : {X_val.shape}")
print(f"Test set           : {X_test.shape}")

print(
    f"\nTraining positive rate   : {np.mean(y_train):.4%}"
)

print(
    f"Validation positive rate : {np.mean(y_val):.4%}"
)

print(
    f"Test positive rate       : {np.mean(y_test):.4%}"
)


# ============================================================
# IMBALANCE RATIO
# ============================================================

positive_count = np.sum(y_train == 1)
negative_count = np.sum(y_train == 0)

scale_pos_weight = negative_count / positive_count

print(
    f"\nscale_pos_weight = {scale_pos_weight:.4f}"
)


# ============================================================
# MODEL FUNCTION
# ============================================================

def create_model(weighted=False):

    params = {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "num_leaves": 35,
        "min_child_samples": 100,

        "subsample": 0.8,
        "subsample_freq": 1,

        "colsample_bytree": 0.8,

        "reg_alpha": 0.5,
        "reg_lambda": 1.0,

        "random_state": RANDOM_STATE,
        "n_jobs": -1,

        "verbosity": -1
    }

    if weighted:
        params["scale_pos_weight"] = scale_pos_weight

    return lgb.LGBMClassifier(**params)


# ============================================================
# METRICS
# ============================================================

def evaluate_model(
    y_true,
    probabilities,
    threshold=0.5
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions
    ).ravel()

    return {

        "roc_auc":
            roc_auc_score(
                y_true,
                probabilities
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probabilities
            ),

        "log_loss":
            log_loss(
                y_true,
                probabilities
            ),

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "true_positive":
            tp,

        "false_positive":
            fp,

        "true_negative":
            tn,

        "false_negative":
            fn
    }


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_f1_threshold(
    y_true,
    probabilities
):

    thresholds = np.linspace(
        0.01,
        0.99,
        197
    )

    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


# ============================================================
# EXPERIMENTS
# ============================================================

experiments = {

    "raw_lightgbm": {
        "smote": False,
        "weighted": False
    },

    "weighted_lightgbm": {
        "smote": False,
        "weighted": True
    },

    "smote_lightgbm": {
        "smote": True,
        "weighted": False
    },

    "smote_weighted_lightgbm": {
        "smote": True,
        "weighted": True
    }
}


results = []

models = {}


# ============================================================
# RUN EXPERIMENTS
# ============================================================

for name, config in experiments.items():

    print("\n")
    print("=" * 80)
    print(f"EXPERIMENT: {name}")
    print("=" * 80)

    X_train_current = X_train.copy()
    y_train_current = y_train.copy()

    # --------------------------------------------------------
    # SMOTE
    # --------------------------------------------------------

    if config["smote"]:

        print("\nApplying SMOTE...")

        smote = SMOTE(
            random_state=RANDOM_STATE
        )

        X_train_current, y_train_current = (
            smote.fit_resample(
                X_train_current,
                y_train_current
            )
        )

        print(
            f"After SMOTE: {X_train_current.shape}"
        )

        print(
            f"Positive rate: "
            f"{np.mean(y_train_current):.4%}"
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = create_model(
        weighted=config["weighted"]
    )

    print("\nTraining...")

    model.fit(
        X_train_current,
        y_train_current,

        eval_set=[
            (X_val, y_val)
        ],

        eval_metric="auc",

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=75,
                verbose=False
            )
        ]
    )

    models[name] = model

    print(
        f"Best iteration: "
        f"{model.best_iteration_}"
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    train_prob = model.predict_proba(
        X_train_current
    )[:, 1]

    val_prob = model.predict_proba(
        X_val
    )[:, 1]

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    train_metrics = evaluate_model(
        y_train_current,
        train_prob
    )

    val_metrics = evaluate_model(
        y_val,
        val_prob
    )

    # --------------------------------------------------------
    # BEST F1 THRESHOLD
    # --------------------------------------------------------

    best_threshold, best_f1 = (
        find_best_f1_threshold(
            y_val,
            val_prob
        )
    )

    threshold_metrics = evaluate_model(
        y_val,
        val_prob,
        best_threshold
    )

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append({

        "model":
            name,

        "smote":
            config["smote"],

        "weighted":
            config["weighted"],

        "train_samples":
            len(y_train_current),

        "best_iteration":
            model.best_iteration_,

        "train_auc":
            train_metrics["roc_auc"],

        "validation_auc":
            val_metrics["roc_auc"],

        "auc_gap":
            train_metrics["roc_auc"]
            -
            val_metrics["roc_auc"],

        "train_pr_auc":
            train_metrics["pr_auc"],

        "validation_pr_auc":
            val_metrics["pr_auc"],

        "validation_log_loss":
            val_metrics["log_loss"],

        "best_f1_threshold":
            best_threshold,

        "best_f1":
            threshold_metrics["f1"],

        "precision_at_best_f1":
            threshold_metrics["precision"],

        "recall_at_best_f1":
            threshold_metrics["recall"]
    })

    print("\nValidation Results")

    print(
        f"ROC-AUC : "
        f"{val_metrics['roc_auc']:.6f}"
    )

    print(
        f"PR-AUC  : "
        f"{val_metrics['pr_auc']:.6f}"
    )

    print(
        f"LogLoss : "
        f"{val_metrics['log_loss']:.6f}"
    )

    print(
        f"Best F1 : "
        f"{threshold_metrics['f1']:.6f}"
    )

    print(
        f"Threshold : "
        f"{best_threshold:.4f}"
    )

    print(
        f"Precision : "
        f"{threshold_metrics['precision']:.6f}"
    )

    print(
        f"Recall : "
        f"{threshold_metrics['recall']:.6f}"
    )


# ============================================================
# COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "validation_auc",
    ascending=False
)

results_path = (
    f"{REPORT_DIR}/lightgbm_strategy_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


print("\n")
print("=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

display_columns = [
    "model",
    "validation_auc",
    "validation_pr_auc",
    "validation_log_loss",
    "auc_gap",
    "best_f1",
    "best_f1_threshold",
    "precision_at_best_f1",
    "recall_at_best_f1"
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)


# ============================================================
# DETERMINE DIRECTION
# ============================================================

best_auc_row = results_df.iloc[0]

best_auc_model = best_auc_row["model"]
best_auc = best_auc_row["validation_auc"]


raw_auc = results_df.loc[
    results_df["model"] == "raw_lightgbm",
    "validation_auc"
].iloc[0]

weighted_auc = results_df.loc[
    results_df["model"] == "weighted_lightgbm",
    "validation_auc"
].iloc[0]

smote_auc = results_df.loc[
    results_df["model"] == "smote_lightgbm",
    "validation_auc"
].iloc[0]

smote_weighted_auc = results_df.loc[
    results_df["model"] == "smote_weighted_lightgbm",
    "validation_auc"
].iloc[0]


# ============================================================
# RECOMMENDATION
# ============================================================

recommendations = []

recommendations.append(
    f"Best validation ROC-AUC model: "
    f"{best_auc_model} ({best_auc:.6f})"
)


# SMOTE analysis

smote_difference = smote_auc - raw_auc

if smote_difference > 0.005:

    recommendations.append(
        f"SMOTE appears beneficial: "
        f"{smote_difference:+.6f} ROC-AUC versus raw data."
    )

elif smote_difference < -0.005:

    recommendations.append(
        f"SMOTE appears harmful: "
        f"{smote_difference:+.6f} ROC-AUC versus raw data. "
        f"Prioritize non-SMOTE approaches."
    )

else:

    recommendations.append(
        f"SMOTE has little effect: "
        f"{smote_difference:+.6f} ROC-AUC versus raw data."
    )


# Class weighting analysis

weight_difference = weighted_auc - raw_auc

if weight_difference > 0.005:

    recommendations.append(
        f"Class weighting improves ranking: "
        f"{weight_difference:+.6f} ROC-AUC."
    )

elif weight_difference < -0.005:

    recommendations.append(
        f"Class weighting reduces ranking quality: "
        f"{weight_difference:+.6f} ROC-AUC."
    )

else:

    recommendations.append(
        f"Class weighting has little effect on ROC-AUC: "
        f"{weight_difference:+.6f}."
    )


# Overfitting

auc_gap = best_auc_row["auc_gap"]

if auc_gap > 0.10:

    recommendations.append(
        "High overfitting detected. Next step: stronger "
        "regularization and cross-validation."
    )

elif auc_gap > 0.05:

    recommendations.append(
        "Moderate overfitting detected. Next step: "
        "regularization and cross-validation."
    )

else:

    recommendations.append(
        "No severe overfitting detected from the AUC gap."
    )


# Final direction

if best_auc_model == "raw_lightgbm":

    recommendations.append(
        "NEXT DIRECTION: use the raw-data pipeline. "
        "Prioritize feature engineering, CatBoost/XGBoost "
        "comparison, and OOF ensembling."
    )

elif best_auc_model == "weighted_lightgbm":

    recommendations.append(
        "NEXT DIRECTION: use class weighting. "
        "Tune scale_pos_weight and investigate "
        "feature engineering."
    )

elif best_auc_model == "smote_lightgbm":

    recommendations.append(
        "NEXT DIRECTION: SMOTE is currently promising. "
        "Test SMOTENC if categorical features exist, "
        "then tune LightGBM."
    )

else:

    recommendations.append(
        "NEXT DIRECTION: SMOTE + class weighting is currently "
        "best. Validate with cross-validation before committing."
    )


# ============================================================
# SAVE DIRECTION REPORT
# ============================================================

direction_path = (
    f"{REPORT_DIR}/lightgbm_direction_report.txt"
)

with open(
    direction_path,
    "w"
) as f:

    f.write(
        "LIGHTGBM STRATEGY DIRECTION REPORT\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
        + "=" * 80
        + "\n"
    )

    f.write(
        "RECOMMENDATIONS\n"
    )

    f.write(
        "=" * 80
        + "\n\n"
    )

    for i, recommendation in enumerate(
        recommendations,
        1
    ):

        f.write(
            f"{i}. {recommendation}\n\n"
        )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================
#
# IMPORTANT:
# We only evaluate the test set here.
# It is NOT used for early stopping or
# model/threshold selection.
#
# ============================================================

print("\n")
print("=" * 80)
print("TEST SET CHECK — BEST VALIDATION MODEL")
print("=" * 80)

best_model = models[
    best_auc_model
]

test_prob = best_model.predict_proba(
    X_test
)[:, 1]

test_auc = roc_auc_score(
    y_test,
    test_prob
)

test_pr_auc = average_precision_score(
    y_test,
    test_prob
)

print(
    f"Test ROC-AUC : {test_auc:.6f}"
)

print(
    f"Test PR-AUC  : {test_pr_auc:.6f}"
)

print(
    "\nIMPORTANT: Test results are reported "
    "for reference only."
)

print(
    "Do NOT tune the model based on this test score."
)


# ============================================================
# SAVE TEST RESULT
# ============================================================

test_result = pd.DataFrame([{

    "selected_model":
        best_auc_model,

    "validation_auc":
        best_auc,

    "test_auc":
        test_auc,

    "validation_pr_auc":
        best_auc_row["validation_pr_auc"],

    "test_pr_auc":
        test_pr_auc,

    "best_iteration":
        best_model.best_iteration_
}])

test_result.to_csv(
    f"{REPORT_DIR}/selected_model_test_result.csv",
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 80)
print("FILES CREATED")
print("=" * 80)

print(
    f"\n{results_path}"
)

print(
    f"{direction_path}"
)

print(
    f"{REPORT_DIR}/selected_model_test_result.csv"
)

print("\n")
print("=" * 80)
print("DIRECTION")
print("=" * 80)

for i, recommendation in enumerate(
    recommendations,
    1
):

    print(
        f"\n{i}. {recommendation}"
    )

print("\nExperiment complete.")