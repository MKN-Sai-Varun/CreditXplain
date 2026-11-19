"""
Final production model training script.
- Trains LightGBM on Feature Engineering V2 features
- Excludes gender features per ethical policy
- Uses validation split for early stopping and threshold selection
- Saves model, model card, and band analysis
- Optional --ablation flag to verify gender exclusion impact

Run: python3 src/final_model.py [--ablation]
"""
import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, precision_recall_curve

from model_params import LGBM_PARAMS, EARLY_STOPPING_ROUNDS, EXCLUDED_FEATURES, GENDER_PATTERNS
from decision import decide

parser = argparse.ArgumentParser()
parser.add_argument("--ablation", action="store_true", help="Run gender ablation study")
args = parser.parse_args()

for d in ("models", "reports", "data/processed"):
    os.makedirs(d, exist_ok=True)

print("=" * 80)
print("FINAL MODEL TRAINING")
print("=" * 80)

# Load data
X = pd.read_csv("data/processed/X_train_fe_v2.csv")
X_test = pd.read_csv("data/processed/X_test_fe_v2.csv")
y = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

print(f"\nOriginal train: {X.shape}")
print(f"Original test: {X_test.shape}")

# Remove excluded features
excluded = []
for feat in EXCLUDED_FEATURES:
    if feat in X.columns:
        excluded.append(feat)
        X = X.drop(columns=[feat])
        X_test = X_test.drop(columns=[feat])

# Remove gender features
gender_cols = [c for c in X.columns if any(p in c for p in GENDER_PATTERNS)]
if gender_cols:
    print(f"\nExcluding gender features: {gender_cols}")
    excluded.extend(gender_cols)
    X = X.drop(columns=gender_cols)
    X_test = X_test.drop(columns=gender_cols)

print(f"\nExcluded features: {excluded}")
print(f"Final train: {X.shape}")
print(f"Final test: {X_test.shape}")

# Train/validation split (80/20 of training data)
X_fit, X_val, y_fit, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nFit set: {X_fit.shape}, positive rate: {y_fit.mean():.4f}")
print(f"Val set: {X_val.shape}, positive rate: {y_val.mean():.4f}")

# Train model
train_data = lgb.Dataset(X_fit, label=y_fit)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

model = lgb.train(
    LGBM_PARAMS,
    train_data,
    num_boost_round=2000,
    valid_sets=[val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)

print(f"\nBest iteration: {model.best_iteration}")

# Predict on validation set (for threshold selection)
val_probs = model.predict(X_val, num_iteration=model.best_iteration)
val_auc = roc_auc_score(y_val, val_probs)

print(f"\nValidation ROC-AUC: {val_auc:.6f}")

# Select thresholds on validation set
# Strategy: Balanced three-band system - aim for ~40% approve, ~40% review, ~20% reject

# Approve threshold: Bottom 40% with acceptable risk (≤3% default rate)
for percentile in range(35, 50, 2):
    threshold_candidate = np.percentile(val_probs, percentile)
    mask = val_probs < threshold_candidate
    if mask.sum() > 0:
        default_rate = y_val[mask].mean()
        if default_rate <= 0.03:  # Target max 3% default in approve band
            approve_below = threshold_candidate
            break
else:
    # Fallback: 40th percentile
    approve_below = np.percentile(val_probs, 40)

# Reject threshold: Top 20% highest risk (80th percentile)
reject_from = np.percentile(val_probs, 80)

# Verify reject band has meaningfully higher risk (at least 3x base rate)
reject_mask = val_probs >= reject_from
if reject_mask.sum() > 0:
    reject_default_rate = y_val[reject_mask].mean()
    # If reject band doesn't have at least 20% default rate, tighten
    if reject_default_rate < 0.20:
        reject_from = np.percentile(val_probs, 85)

print(f"\nSelected thresholds (from validation set):")
print(f"  Approve below: {approve_below:.4f}")
print(f"  Reject from:   {reject_from:.4f}")

# Report validation band distribution
val_decisions = decide(val_probs, approve_below, reject_from)
print(f"\nValidation set band distribution:")
for band in ["approve", "review", "reject"]:
    mask = val_decisions == band
    count = mask.sum()
    if count > 0:
        rate = y_val[mask].mean()
        pct = count / len(val_decisions) * 100
        print(f"  {band.capitalize()}: {count:,} ({pct:.1f}%) - Default rate: {rate:.2%}")
    else:
        print(f"  {band.capitalize()}: 0 (0.0%) - Default rate: N/A")

# Score test set once
test_probs = model.predict(X_test, num_iteration=model.best_iteration)
test_auc = roc_auc_score(y_test, test_probs)
test_pr_auc = average_precision_score(y_test, test_probs)
test_logloss = log_loss(y_test, test_probs)

print(f"\nTest set performance:")
print(f"  ROC-AUC: {test_auc:.6f}")
print(f"  PR-AUC:  {test_pr_auc:.6f}")
print(f"  LogLoss: {test_logloss:.6f}")

# Band analysis on test set
test_decisions = decide(test_probs, approve_below, reject_from)
band_stats = []

for band in ["approve", "review", "reject"]:
    mask = test_decisions == band
    count = mask.sum()
    if count > 0:
        rate = y_test[mask].mean()
        avg_prob = test_probs[mask].mean()
    else:
        rate = 0.0
        avg_prob = 0.0
    
    band_stats.append({
        "band": band,
        "count": count,
        "percentage": count / len(test_decisions) * 100,
        "actual_default_rate": rate,
        "avg_predicted_prob": avg_prob
    })

bands_df = pd.DataFrame(band_stats)
print("\n" + "=" * 80)
print("DECISION BAND ANALYSIS (Test Set)")
print("=" * 80)
print(bands_df.to_string(index=False))

# Save model
joblib.dump(model, "models/final_lgbm.pkl")
print("\nSaved model: models/final_lgbm.pkl")

# Save model card
model_card = {
    "model_type": "LightGBM",
    "features_used": X.columns.tolist(),
    "features_excluded": excluded,
    "training_samples": len(X_fit),
    "validation_samples": len(X_val),
    "test_samples": len(X_test),
    "best_iteration": model.best_iteration,
    "validation_roc_auc": float(val_auc),
    "test_roc_auc": float(test_auc),
    "test_pr_auc": float(test_pr_auc),
    "test_logloss": float(test_logloss),
    "approve_threshold": float(approve_below),
    "reject_threshold": float(reject_from),
    "hyperparameters": LGBM_PARAMS,
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS
}

with open("models/final_model_card.json", "w") as f:
    json.dump(model_card, f, indent=2)

print("Saved model card: models/final_model_card.json")

# Save band analysis
bands_df.to_csv("reports/final_model_bands.csv", index=False)
print("Saved bands: reports/final_model_bands.csv")

# Save audit test set (for fairness analysis in Phase 3)
audit_df = X_test.copy()
audit_df["predicted_probability"] = test_probs
audit_df["decision"] = test_decisions
audit_df["actual_default"] = y_test.values

audit_df.to_csv("data/processed/audit_test.csv", index=False)
print("Saved audit data: data/processed/audit_test.csv")

# Optional: Gender ablation study
if args.ablation:
    print("\n" + "=" * 80)
    print("GENDER ABLATION STUDY")
    print("=" * 80)
    
    # Reload original features with gender
    X_with_gender = pd.read_csv("data/processed/X_train_fe_v2.csv")
    X_test_with_gender = pd.read_csv("data/processed/X_test_fe_v2.csv")
    
    # Remove only DAYS_BIRTH (keep gender this time)
    for feat in EXCLUDED_FEATURES:
        if feat in X_with_gender.columns:
            X_with_gender = X_with_gender.drop(columns=[feat])
            X_test_with_gender = X_test_with_gender.drop(columns=[feat])
    
    X_fit_g, X_val_g, y_fit_g, y_val_g = train_test_split(
        X_with_gender, y, test_size=0.2, stratify=y, random_state=42
    )
    
    train_g = lgb.Dataset(X_fit_g, label=y_fit_g)
    val_g = lgb.Dataset(X_val_g, label=y_val_g, reference=train_g)
    
    print("\nTraining model WITH gender features...")
    model_with_gender = lgb.train(
        LGBM_PARAMS,
        train_g,
        num_boost_round=2000,
        valid_sets=[val_g],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    
    test_probs_g = model_with_gender.predict(X_test_with_gender, num_iteration=model_with_gender.best_iteration)
    test_auc_g = roc_auc_score(y_test, test_probs_g)
    
    ablation_results = pd.DataFrame([
        {
            "model": "without_gender",
            "test_roc_auc": test_auc,
            "features_count": len(X.columns)
        },
        {
            "model": "with_gender",
            "test_roc_auc": test_auc_g,
            "features_count": len(X_with_gender.columns)
        }
    ])
    
    ablation_results["auc_difference"] = ablation_results["test_roc_auc"].diff()
    
    print("\n" + "=" * 80)
    print("ABLATION RESULTS")
    print("=" * 80)
    print(ablation_results.to_string(index=False))
    
    ablation_results.to_csv("reports/gender_ablation.csv", index=False)
    print("\nSaved: reports/gender_ablation.csv")

print("\n" + "=" * 80)
print("FINAL MODEL TRAINING COMPLETE")
print("=" * 80)
