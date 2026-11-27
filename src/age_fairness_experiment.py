"""
Does age need to be a model feature? (Phase 5.5a, KNOWN_ISSUES #16, Q5)

Compares, on the held-out test set:
  A  current       the final model (age kept, gender excluded)
  B  no_age        A without AGE_YEARS, AGE_SQUARED, AGE_CUBED
  C  age_free      B without the features calculated from age
                   (EMPLOYMENT_AGE_RATIO, AGE_MINUS_EMPLOYMENT, REGISTRATION_AGE_RATIO)
  D  band_cutoffs  the current model with approve and reject cutoffs set per age
                   band so each band has the overall approve and reject rates
                   (reference only: it uses age at decision time)

B and C repeat src/final_model.py exactly: same parameters, fit/validation
split, early stopping and cutoff rule. For each variant it reports accuracy,
the bands, and per age band the actual default rate, approve rate, AUC, the
approve rate among applicants who repaid (equal opportunity) the default
rate among the approved, and the mean predicted risk (calibration). A proxy check measures how well age can still be
predicted from the age-free features.

Run: python3 -m src.age_fairness_experiment [--threads 2]
Writes reports/age_fairness_experiment.json and models/age_free_lgbm.pkl (gitignored).
"""
import argparse
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.decision import decide, select_cutoffs
from src.model_params import EARLY_STOPPING_ROUNDS, LGBM_PARAMS

warnings.filterwarnings("ignore")
REPORT = Path("reports/age_fairness_experiment.json")
DIRECT_AGE = ["AGE_YEARS", "AGE_SQUARED", "AGE_CUBED"]
FROM_AGE = ["EMPLOYMENT_AGE_RATIO", "AGE_MINUS_EMPLOYMENT", "REGISTRATION_AGE_RATIO"]
AGE_BINS = [0, 25, 35, 45, 55, 100]
AGE_LABELS = ["<25", "25-35", "35-45", "45-55", "55+"]   # the bands of src/fairness_audit.py


def age_bands(age_years):
    return pd.cut(pd.Series(age_years), bins=AGE_BINS, labels=AGE_LABELS).astype(str).values


def group_table(y, probs, bands, groups):
    """Per group: size, actual default rate, decision rates, AUC, equal-opportunity rate, default rate when approved."""
    y, probs, bands, groups = map(np.asarray, (y, probs, bands, groups))
    rows = []
    for g in [x for x in AGE_LABELS if x in set(groups)] or sorted(set(groups)):
        m = groups == g
        yg, bg = y[m], bands[m]
        approved = bg == "approve"
        repaid = yg == 0
        rows.append({
            "group": g,
            "n": int(m.sum()),
            "actual_default_rate": round(float(yg.mean()), 4),
            "mean_predicted_risk": round(float(probs[m].mean()), 4),   # calibration: close to the actual rate?
            "approve_rate": round(float(approved.mean()), 4),
            "review_rate": round(float((bg == "review").mean()), 4),
            "reject_rate": round(float((bg == "reject").mean()), 4),
            "auc": round(float(roc_auc_score(yg, probs[m])), 4) if 0 < yg.sum() < len(yg) else None,
            "approve_rate_among_repaid": round(float(approved[repaid].mean()), 4) if repaid.any() else None,
            "default_rate_among_approved": round(float(yg[approved].mean()), 4) if approved.any() else None,
        })
    return rows


def gaps(rows):
    """Largest difference between groups: approve rate (demographic parity) and approve rate among repaid."""
    def spread(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(max(vals) - min(vals), 4) if vals else None
    return {"demographic_parity_difference": spread("approve_rate"),
            "equal_opportunity_difference": spread("approve_rate_among_repaid"),
            "actual_default_rate_spread": spread("actual_default_rate")}


def per_group_cutoffs(val_probs, val_groups, approve_rate, reject_rate):
    """Cutoffs per group so that, on validation, each group is approved and rejected at the given rates."""
    out = {}
    for g in set(val_groups):
        p = np.asarray(val_probs)[np.asarray(val_groups) == g]
        out[g] = (float(np.quantile(p, approve_rate)), float(np.quantile(p, 1 - reject_rate)))
    return out


def decide_per_group(probs, groups, cutoffs):
    bands = np.empty(len(probs), dtype=object)
    for g, (a, r) in cutoffs.items():
        m = np.asarray(groups) == g
        bands[m] = decide(np.asarray(probs)[m], a, r)
    return bands


def summary(name, y, probs, bands, groups, cutoffs, extra=None):
    rows = group_table(y, probs, bands, groups)
    approved = bands == "approve"
    return {
        "variant": name,
        "test_roc_auc": round(float(roc_auc_score(y, probs)), 4),
        "test_pr_auc": round(float(average_precision_score(y, probs)), 4),
        "cutoffs": cutoffs,
        "approve_rate": round(float(approved.mean()), 4),
        "review_rate": round(float((bands == "review").mean()), 4),
        "reject_rate": round(float((bands == "reject").mean()), 4),
        "default_rate_among_approved": round(float(np.asarray(y)[approved].mean()), 4),
        **gaps(rows),
        "by_age_band": rows,
        **(extra or {}),
    }


def train(features, X_fit, y_fit, X_val, y_val, threads):
    params = {**LGBM_PARAMS, "n_jobs": threads}
    fit = lgb.Dataset(X_fit[features], label=y_fit)
    val = lgb.Dataset(X_val[features], label=y_val, reference=fit)
    return lgb.train(params, fit, num_boost_round=2000, valid_sets=[val],
                     callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])


def proxy_check(features, X_fit, age_fit, X_test, age_test, threads, n_fit=60000):
    """How well the remaining features predict age (R², mean absolute error in years)."""
    idx = np.random.default_rng(0).choice(len(X_fit), min(n_fit, len(X_fit)), replace=False)
    reg = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63, n_jobs=threads, verbosity=-1,
                            random_state=0)
    reg.fit(X_fit[features].iloc[idx], age_fit[idx])
    pred = reg.predict(X_test[features])
    return {"r2": round(float(r2_score(age_test, pred)), 4),
            "mean_absolute_error_years": round(float(mean_absolute_error(age_test, pred)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    try:
        os.nice(10)
    except OSError:
        pass

    card = json.load(open("models/final_model_card.json"))
    feats = card["features_used"]
    X = pd.read_csv("data/processed/X_train_fe_v2.csv")
    y = pd.read_csv("data/processed/y_train.csv").squeeze()
    X_test = pd.read_csv("data/processed/X_test_fe_v2.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze().values
    X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    groups_test = age_bands(X_test["AGE_YEARS"].values)
    groups_val = age_bands(X_val["AGE_YEARS"].values)

    results = {}

    # A: the saved final model
    model = joblib.load("models/final_lgbm.pkl")
    probs_a = model.predict(X_test[feats])
    a_cut = (card["approve_threshold"], card["reject_threshold"])
    bands_a = decide(probs_a, *a_cut)
    results["A_current"] = summary("current", y_test, probs_a, bands_a, groups_test,
                                   {"approve_below": a_cut[0], "reject_from": a_cut[1]},
                                   {"n_features": len(feats)})
    print("A done", flush=True)

    # B and C: retrained without age
    for key, name, drop in (("B_no_age", "no_age", DIRECT_AGE), ("C_age_free", "age_free", DIRECT_AGE + FROM_AGE)):
        f = [c for c in feats if c not in drop]
        t = time.time()
        m = train(f, X_fit, y_fit, X_val, y_val, args.threads)
        val_probs = m.predict(X_val[f], num_iteration=m.best_iteration)
        a, r = select_cutoffs(val_probs, y_val)
        probs = m.predict(X_test[f], num_iteration=m.best_iteration)
        results[key] = summary(name, y_test, probs, decide(probs, a, r), groups_test,
                               {"approve_below": float(a), "reject_from": float(r)},
                               {"n_features": len(f), "dropped": [c for c in drop if c in feats],
                                "best_iteration": int(m.best_iteration),
                                "train_seconds": round(time.time() - t)})
        if key == "C_age_free":
            joblib.dump(m, "models/age_free_lgbm.pkl")
            age_free = f
        print(f"{key} done ({time.time() - t:.0f} s)", flush=True)

    # D: current model, cutoffs per age band (reference only)
    val_probs_a = model.predict(X_val[feats])
    val_bands = decide(val_probs_a, *a_cut)
    cut_d = per_group_cutoffs(val_probs_a, groups_val, float((val_bands == "approve").mean()),
                              float((val_bands == "reject").mean()))
    results["D_band_cutoffs"] = summary("band_cutoffs", y_test, probs_a,
                                        decide_per_group(probs_a, groups_test, cut_d), groups_test,
                                        {g: {"approve_below": round(a, 4), "reject_from": round(r, 4)}
                                         for g, (a, r) in sorted(cut_d.items())},
                                        {"note": "uses age at decision time; reference only"})
    print("D done", flush=True)

    proxy = proxy_check(age_free, X_fit, X_fit["AGE_YEARS"].values, X_test, X_test["AGE_YEARS"].values,
                        args.threads)
    print("proxy check done", flush=True)

    out = {
        "test_applicants": int(len(y_test)),
        "age_bands": AGE_LABELS,
        "variants": results,
        "age_predictable_from_age_free_features": proxy,
        "notes": {
            "demographic_parity_difference": "largest minus smallest approve rate across age bands",
            "equal_opportunity_difference": "same, among applicants who repaid (did not default)",
            "actual_default_rate_spread": "difference in real default rates between age bands, from the data",
        },
    }
    REPORT.write_text(json.dumps(out, indent=2))
    for k, v in results.items():
        print(f"{k:16s} AUC {v['test_roc_auc']:.4f}  approve {v['approve_rate']:.3f}  "
              f"DPD {v['demographic_parity_difference']:.3f}  EOD {v['equal_opportunity_difference']:.3f}  "
              f"default|approved {v['default_rate_among_approved']:.4f}")
    print("proxy:", proxy)


if __name__ == "__main__":
    main()
