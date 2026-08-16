"""
Phase 2: Baseline models - Logistic Regression, Random Forest, XGBoost, LightGBM
Run: python3 src/train_baseline.py
"""
import re
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

DATA_PATH = "data/processed/app_train_clean.csv"

df = pd.read_csv(DATA_PATH)

DROP_COLS = ["TARGET", "SK_ID_CURR", "SIMULATED_SUBMITTED_AT"]
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
X = pd.get_dummies(X, drop_first=True)
y = df["TARGET"]

# LightGBM rejects special JSON characters in feature names (commas, brackets,
# etc. show up here from category values via get_dummies). Sanitize.
X.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", str(c)) for c in X.columns]
# sanitizing can create duplicate names - make them unique
if X.columns.duplicated().any():
    seen = {}
    new_cols = []
    for c in X.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    X.columns = new_cols

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)


def best_f1_threshold(y_true, probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    idx = f1s.argmax()
    thresh = thresholds[idx] if idx < len(thresholds) else 0.5
    return thresh, precisions[idx], recalls[idx], f1s[idx]

# Logistic Regression needs scaled features - unscaled AMT_* columns (in the
# hundreds of thousands) vs binary flags (0/1) cause overflow during optimization.
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

models_scaled = {"logistic_regression": LogisticRegression(max_iter=2000)}
models_unscaled = {
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "xgboost": XGBClassifier(eval_metric="logloss", random_state=42),
    "lightgbm": LGBMClassifier(random_state=42),
}

results = {}

for name, model in models_scaled.items():
    print(f"Training {name}...")
    model.fit(X_train_res_scaled, y_train_res)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary")
    thresh, t_prec, t_rec, t_f1 = best_f1_threshold(y_test, probs)
    results[name] = {
        "accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "roc_auc": auc,
        "best_threshold": thresh, "thresh_precision": t_prec, "thresh_recall": t_rec, "thresh_f1": t_f1,
    }
    joblib.dump(model, f"models/{name}.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

for name, model in models_unscaled.items():
    print(f"Training {name}...")
    model.fit(X_train_res, y_train_res)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary")
    thresh, t_prec, t_rec, t_f1 = best_f1_threshold(y_test, probs)
    results[name] = {
        "accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "roc_auc": auc,
        "best_threshold": thresh, "thresh_precision": t_prec, "thresh_recall": t_rec, "thresh_f1": t_f1,
    }
    joblib.dump(model, f"models/{name}.pkl")

results_df = pd.DataFrame(results).T
print(results_df)
results_df.to_csv("reports/baseline_model_comparison.csv")