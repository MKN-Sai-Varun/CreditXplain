import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score, log_loss


LGBM_PATH = "reports/lgbm_fe_v2_oof.csv"
CATBOOST_PATH = "reports/catboost_fe_v2_oof.csv"

lgbm = pd.read_csv(LGBM_PATH)
catboost = pd.read_csv(CATBOOST_PATH)

y = lgbm["target"].values
lgbm_pred = lgbm["oof_prediction"].values
cat_pred = catboost["prediction"].values

print("=" * 80)
print("LIGHTGBM + CATBOOST OOF ENSEMBLE")
print("=" * 80)

print("\nIndividual Models")
print(f"LightGBM AUC : {roc_auc_score(y, lgbm_pred):.6f}")
print(f"CatBoost AUC : {roc_auc_score(y, cat_pred):.6f}")

print("\nPrediction Correlation")
print(np.corrcoef(lgbm_pred, cat_pred)[0, 1])

results = []

for w in np.arange(0.0, 1.01, 0.05):

    ensemble_pred = (
        w * lgbm_pred
        + (1.0 - w) * cat_pred
    )

    auc = roc_auc_score(y, ensemble_pred)
    pr_auc = average_precision_score(y, ensemble_pred)
    ll = log_loss(y, ensemble_pred)

    results.append({
        "lgbm_weight": w,
        "catboost_weight": 1.0 - w,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "logloss": ll
    })


results_df = pd.DataFrame(results)

best_idx = results_df["roc_auc"].idxmax()
best = results_df.loc[best_idx]

print("\n" + "=" * 80)
print("BLEND RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        formatters={
            "lgbm_weight": "{:.2f}".format,
            "catboost_weight": "{:.2f}".format,
            "roc_auc": "{:.6f}".format,
            "pr_auc": "{:.6f}".format,
            "logloss": "{:.6f}".format
        }
    )
)

print("\n" + "=" * 80)
print("BEST BLEND")
print("=" * 80)

print(f"LightGBM weight : {best['lgbm_weight']:.2f}")
print(f"CatBoost weight : {best['catboost_weight']:.2f}")
print(f"ROC-AUC         : {best['roc_auc']:.6f}")
print(f"PR-AUC          : {best['pr_auc']:.6f}")
print(f"LogLoss         : {best['logloss']:.6f}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print(f"Best LightGBM : 0.771558")
print(f"Best Ensemble : {best['roc_auc']:.6f}")
print(f"Improvement   : {best['roc_auc'] - 0.771558:+.6f}")

results_df.to_csv(
    "reports/lgbm_catboost_blend_results.csv",
    index=False
)

ensemble_pred = (
    best["lgbm_weight"] * lgbm_pred
    + best["catboost_weight"] * cat_pred
)

pd.DataFrame({
    "prediction": ensemble_pred,
    "target": y
}).to_csv(
    "reports/lgbm_catboost_ensemble_oof.csv",
    index=False
)

print("\nSaved:")
print("reports/lgbm_catboost_blend_results.csv")
print("reports/lgbm_catboost_ensemble_oof.csv")

print("\nDONE")