"""
SHAP vs LIME agreement on held-out applicants, for two LIME set-ups:
  lime_default : LIME on all features, continuous, sampled around the
                 training mean (the set-up used before Phase 4.7)
  lime_local   : src/lime_local.py (top-gain features, sampled per applicant)

Both are fitted to the model's log-odds, the scale SHAP uses.
Run: python3 -m src.explanation_eval [--per-band 50] [--seeds 5]
"""
import argparse
import json
import time
import warnings
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

from src.decision import decide
from src.lime_local import LocalLime

warnings.filterwarnings("ignore")   # numpy/Accelerate matmul noise, see KNOWN_ISSUES #53


def top(v, k):
    return set(np.argsort(-np.abs(v))[:k])


def agreement(lime_w, shap_v, k_rank=25):
    """lime_w and shap_v are full-length vectors over the model's features."""
    ref = np.argsort(-np.abs(shap_v))[:k_rank]
    top5 = list(top(shap_v, 5))
    return {
        "top5_overlap": len(top(lime_w, 5) & top(shap_v, 5)) / 5,
        "top10_overlap": len(top(lime_w, 10) & top(shap_v, 10)) / 10,
        "spearman": spearmanr(lime_w[ref], shap_v[ref]).correlation,
        "sign_agreement_top5": float(np.mean(np.sign(lime_w[top5]) == np.sign(shap_v[top5]))),
    }


def jaccard(a, b):
    return len(a & b) / len(a | b)


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-band", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    card = json.load(open("models/final_model_card.json"))
    feats = card["features_used"]
    model = joblib.load("models/final_lgbm.pkl")
    background = pd.read_csv("data/processed/X_train_fe_v2.csv", nrows=5000)[feats].astype(float)
    X = pd.read_csv("data/processed/X_test_fe_v2.csv")[feats].astype(float)

    bands = decide(model.predict(X), card["approve_threshold"], card["reject_threshold"])
    rng = np.random.default_rng(42)
    pick = np.concatenate([
        rng.choice(np.where(bands == b)[0], args.per_band, replace=False)
        for b in ("approve", "review", "reject")
    ])
    Xs = X.iloc[pick].reset_index(drop=True)
    sv = shap.TreeExplainer(model).shap_values(Xs)
    sv = np.asarray(sv[-1] if isinstance(sv, list) else sv)
    # a different applicant's SHAP vector, for the locality check
    other = np.roll(np.arange(len(Xs)), 1)

    raw = lambda Z: model.predict(Z, raw_score=True)
    default = LimeTabularExplainer(background.values, feature_names=feats, mode="regression",
                                   discretize_continuous=False, random_state=0)
    local = LocalLime(model, feats, background)

    rows = []
    for i in range(len(Xs)):
        x = Xs.iloc[i].values
        band = bands[pick[i]]

        t = time.perf_counter()
        e = default.explain_instance(x, raw, num_features=len(feats), num_samples=5000)
        ms_default = 1000 * (time.perf_counter() - t)
        w_default = np.zeros(len(feats))
        for j, w in e.as_map()[1]:
            w_default[j] = w

        runs = []
        for seed in range(args.seeds):
            t = time.perf_counter()
            _, w_sub, score = local.explain(x, seed=seed)
            ms = 1000 * (time.perf_counter() - t)
            w_full = np.zeros(len(feats))
            w_full[local.idx] = w_sub
            runs.append((w_full, score, ms))
        w_local, score_local, ms_local = runs[0]

        for name, w, score, ms in (("lime_default", w_default, e.score, ms_default),
                                   ("lime_local", w_local, score_local, ms_local)):
            own = agreement(w, sv[i])
            shuffled = agreement(w, sv[other[i]])
            rows.append({
                "applicant": int(pick[i]), "band": band, "method": name,
                **own,
                "surrogate_r2": score,
                "locality_gap": own["spearman"] - shuffled["spearman"],
                "seed_stability_top5": (np.mean([jaccard(top(a[0], 5), top(b[0], 5))
                                                 for a, b in combinations(runs, 2)])
                                        if name == "lime_local" else np.nan),
                "latency_ms": ms,
            })
        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{len(Xs)} applicants")

    df = pd.DataFrame(rows)
    metrics = ["top5_overlap", "top10_overlap", "spearman", "sign_agreement_top5",
               "surrogate_r2", "locality_gap", "seed_stability_top5", "latency_ms"]
    summary = {
        "applicants": int(len(Xs)),
        "per_band": args.per_band,
        "seeds": args.seeds,
        "lime_local_features": local.subset,
        "overall": df.groupby("method")[metrics].mean().round(4).to_dict(orient="index"),
        "by_band": {
            m: g.groupby("band")[metrics].mean().round(4).to_dict(orient="index")
            for m, g in df.groupby("method")
        },
    }
    df.to_csv("reports/explanation_consistency.csv", index=False)
    with open("reports/explanation_consistency.json", "w") as f:
        json.dump(clean(summary), f, indent=2, default=float)
    print(json.dumps(clean(summary["overall"]), indent=2))


if __name__ == "__main__":
    main()
