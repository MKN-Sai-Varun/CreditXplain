"""
Diagnostic: check if low recall is a threshold problem, not a model problem.
Run after train_baseline.py: python3 src/threshold_check.py
"""
import pandas as pd
import joblib
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

for name in ["random_forest", "xgboost", "lightgbm"]:
    model = joblib.load(f"models/{name}.pkl")
    probs = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {name} ===")
    print("Confusion matrix @ threshold 0.5:")
    print(confusion_matrix(y_test, (probs >= 0.5).astype(int)))

    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = f1s.argmax()
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    print(f"Best F1 threshold: {best_thresh:.3f} "
          f"(precision={precisions[best_idx]:.3f}, recall={recalls[best_idx]:.3f}, f1={f1s[best_idx]:.3f})")
    print("Confusion matrix @ best threshold:")
    print(confusion_matrix(y_test, (probs >= best_thresh).astype(int)))