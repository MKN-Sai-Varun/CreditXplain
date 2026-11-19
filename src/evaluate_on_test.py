"""
Scores the FE v2 models on the untouched test set.
The blend weight comes from the out-of-fold sweep, never tuned on test.
Run: python3 src/evaluate_on_test.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

y_test = pd.read_csv("data/processed/y_test.csv")["TARGET"].values


def load_pred(path):
    df = pd.read_csv(path)
    if "prediction" in df.columns:
        return df["prediction"].values
    assert df.shape[1] == 1, f"unexpected columns in {path}: {list(df.columns)}"
    return df.iloc[:, 0].values


lgbm = load_pred("reports/lgbm_fe_v2_test.csv")
cat = load_pred("reports/catboost_fe_v2_test.csv")
assert len(lgbm) == len(cat) == len(y_test), (len(lgbm), len(cat), len(y_test))

sweep = pd.read_csv("reports/lgbm_catboost_blend_results.csv")
w = float(sweep.loc[sweep["roc_auc"].idxmax(), "lgbm_weight"])

candidates = {
    "lgbm_fe_v2": lgbm,
    "catboost_fe_v2": cat,
    f"blend_w{w:.2f}": w * lgbm + (1 - w) * cat,
}

rows = []
for name, p in candidates.items():
    auc = roc_auc_score(y_test, p)
    if auc < 0.6:
        print(f"WARNING: {name} AUC {auc:.3f} is near chance - test rows are probably misaligned")
    rows.append({
        "model": name,
        "test_roc_auc": round(auc, 4),
        "test_pr_auc": round(average_precision_score(y_test, p), 4),
        "test_logloss": round(log_loss(y_test, p), 4),
    })

out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv("reports/fe_v2_test_results.csv", index=False)