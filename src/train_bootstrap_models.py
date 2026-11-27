"""
Retrains the final model on bootstrap resamples, to test whether recourse
still holds after the model is retrained.

Each model repeats src/final_model.py exactly (same features, parameters, early
stopping on the same validation split, same cutoff rule), but is fitted on a
resample, with replacement, of the fit split and uses its own seed.

Run: python3 -m src.train_bootstrap_models [--n 10] [--threads 2]
Models 1-5 serve the ensemble-robust recourse search; 6-10 are kept for evaluation only.
Writes models/bootstrap_lgbm_<k>.pkl (gitignored) and reports/bootstrap_models.json.
Already-trained models are kept, so an interrupted run can be resumed.
"""
import argparse
import json
import os
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.decision import select_cutoffs
from src.model_params import EARLY_STOPPING_ROUNDS, LGBM_PARAMS

REPORT = Path("reports/bootstrap_models.json")


def model_path(k):
    return Path(f"models/bootstrap_lgbm_{k}.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="number of bootstrap models")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads for training")
    args = ap.parse_args()
    try:
        os.nice(10)
    except OSError:
        pass

    card = json.load(open("models/final_model_card.json"))
    feats = card["features_used"]
    X = pd.read_csv("data/processed/X_train_fe_v2.csv")[feats]
    y = pd.read_csv("data/processed/y_train.csv").squeeze()
    X_test = pd.read_csv("data/processed/X_test_fe_v2.csv")[feats]
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    # the same fit/validation split as src/final_model.py
    X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    val_data = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

    report = json.load(open(REPORT)) if REPORT.exists() else {"models": []}
    done = {m["k"] for m in report["models"] if model_path(m["k"]).exists()}
    report["models"] = [m for m in report["models"] if m["k"] in done]
    report["final_model"] = {"test_roc_auc": card["test_roc_auc"], "approve_threshold": card["approve_threshold"],
                             "reject_threshold": card["reject_threshold"]}

    for k in range(1, args.n + 1):
        if k in done:
            print(f"model {k}: already trained, kept")
            continue
        t0 = time.time()
        seed = 100 + k
        idx = np.random.default_rng(seed).integers(0, len(X_fit), len(X_fit))
        params = dict(LGBM_PARAMS, random_state=seed, n_jobs=args.threads)
        model = lgb.train(
            params,
            lgb.Dataset(X_fit.iloc[idx], label=y_fit.iloc[idx], reference=val_data),
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        val_probs = model.predict(X_val, num_iteration=model.best_iteration)
        approve_below, reject_from = select_cutoffs(val_probs, y_val)
        test_auc = roc_auc_score(y_test, model.predict(X_test, num_iteration=model.best_iteration))

        joblib.dump(model, model_path(k))
        report["models"].append({
            "k": k,
            "seed": seed,
            "path": str(model_path(k)),
            "best_iteration": int(model.best_iteration),
            "validation_roc_auc": float(roc_auc_score(y_val, val_probs)),
            "test_roc_auc": float(test_auc),
            "approve_threshold": approve_below,
            "reject_threshold": reject_from,
            "unique_rows_share": float(len(np.unique(idx)) / len(idx)),
        })
        report["models"].sort(key=lambda m: m["k"])
        REPORT.write_text(json.dumps(report, indent=2))
        print(f"model {k}: test AUC {test_auc:.4f}, cutoffs {approve_below:.4f} / {reject_from:.4f}, "
              f"{model.best_iteration} trees, {time.time() - t0:.0f} s")

    aucs = [m["test_roc_auc"] for m in report["models"]]
    report["test_roc_auc_mean"] = float(np.mean(aucs))
    report["test_roc_auc_std"] = float(np.std(aucs))
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"\n{len(aucs)} models, test AUC {np.mean(aucs):.4f} ± {np.std(aucs):.4f} "
          f"(final model {card['test_roc_auc']:.4f})")


if __name__ == "__main__":
    main()
