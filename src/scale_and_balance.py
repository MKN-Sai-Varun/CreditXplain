"""
Scaling + SMOTE stage - runs after train_baseline.py.
Drops DAYS_BIRTH (collinear with AGE_YEARS), scales continuous features,
applies SMOTE to training data only.
Run: python3 src/scale_and_balance.py
"""
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# --- Train ---
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

if "DAYS_BIRTH" in X_train.columns:
    X_train = X_train.drop(columns=["DAYS_BIRTH"])

# Determine continuous columns ONCE, from training data only.
# This list is reused for the test set below - never recomputed independently,
# since a column's cardinality could differ slightly between splits and
# scaler.transform() must see the exact same columns it was fit on.
continuous_cols = [col for col in X_train.columns if X_train[col].nunique() > 2]

scaler = StandardScaler()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
joblib.dump(scaler, "models/credit_scaler.pkl")
joblib.dump(continuous_cols, "models/continuous_cols.pkl")

smote = SMOTE(random_state=42)
X_train_modified, y_train_modified = smote.fit_resample(X_train, y_train)

X_train_modified.to_csv("data/processed/X_train_modified.csv", index=False)
y_train_modified.to_csv("data/processed/y_train_modified.csv", index=False)

print(f"Train - original shape: {X_train.shape} | after SMOTE: {X_train_modified.shape}")

# --- Test ---
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

if "DAYS_BIRTH" in X_test.columns:
    X_test = X_test.drop(columns=["DAYS_BIRTH"])

# Reuse the exact same continuous_cols list from training - do NOT recompute.
scaler = joblib.load("models/credit_scaler.pkl")
continuous_cols = joblib.load("models/continuous_cols.pkl")
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

X_test.to_csv("data/processed/X_test_modified.csv", index=False)
y_test.to_csv("data/processed/y_test_modified.csv", index=False)

print(f"Test - features shape: {X_test.shape} | labels shape: {y_test.shape}")