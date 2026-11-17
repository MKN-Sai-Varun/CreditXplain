"""
Scaling stage. Drops DAYS_BIRTH, scales continuous features, saves the scaler.
Class balancing is OFF by default: validation experiments showed SMOTE lowered
ROC-AUC, and imbalance is handled at the decision threshold instead.
Pass --smote to oversample the training set with SMOTENC (binary flags handled correctly).

Run: python3 src/scale_and_balance.py [--smote]
"""
import argparse
import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("--smote", action="store_true", help="oversample training data with SMOTENC")
args = parser.parse_args()

for d in ("models", "data/processed", "reports"):
    os.makedirs(d, exist_ok=True)


def bools_to_int(df):
    return df.astype({c: int for c in df.columns if df[c].dtype == bool})


X_train = bools_to_int(pd.read_csv("data/processed/X_train.csv"))
y_train = pd.read_csv("data/processed/y_train.csv")
X_test = bools_to_int(pd.read_csv("data/processed/X_test.csv"))
y_test = pd.read_csv("data/processed/y_test.csv")

X_train = X_train.drop(columns=["DAYS_BIRTH"], errors="ignore")
X_test = X_test.drop(columns=["DAYS_BIRTH"], errors="ignore")

# Continuous columns are decided once, from training data, and reused for test.
continuous_cols = [c for c in X_train.columns if X_train[c].nunique() > 2]

scaler = StandardScaler()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])
joblib.dump(scaler, "models/credit_scaler.pkl")
joblib.dump(continuous_cols, "models/continuous_cols.pkl")

if args.smote:
    from imblearn.over_sampling import SMOTENC

    cat_mask = [c not in continuous_cols for c in X_train.columns]
    X_train, y_train = SMOTENC(categorical_features=cat_mask, random_state=42).fit_resample(X_train, y_train)

X_train.to_csv("data/processed/X_train_modified.csv", index=False)
y_train.to_csv("data/processed/y_train_modified.csv", index=False)
X_test.to_csv("data/processed/X_test_modified.csv", index=False)
y_test.to_csv("data/processed/y_test_modified.csv", index=False)

print(f"Train: {X_train.shape} (positive rate {y_train['TARGET'].mean():.4f}) | Test: {X_test.shape}")