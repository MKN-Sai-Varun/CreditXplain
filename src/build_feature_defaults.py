"""
Saves per-column medians of the base training features.
Used at inference to fill fields the applicant form does not collect.
Run: python3 src/build_feature_defaults.py
"""
import json
import os
import pandas as pd

X = pd.read_csv("data/processed/X_train.csv")
X = X.astype({c: int for c in X.columns if X[c].dtype == bool})

os.makedirs("models", exist_ok=True)
with open("models/feature_defaults.json", "w") as f:
    json.dump({c: float(v) for c, v in X.median().items()}, f, indent=1)

print(f"Saved {X.shape[1]} defaults -> models/feature_defaults.json")