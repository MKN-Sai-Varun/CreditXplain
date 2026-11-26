"""
Evaluates recourse on held-out applicants who are not approved.

Methods (all limited to the same three levers and the same bounds):
  previous  - the pre-4.7 engine: grid only, no margin, no plausibility check
  proposed  - src/recourse.py defaults: grid + refinement, margin, plausibility
  random    - random search over the same box with a similar evaluation budget,
              same margin and plausibility rules
  dice      - DiCE (random method) on the same levers, scored through the
              same FE v2 path; about 40 s per applicant, so run on a subset (--dice N)
  ablation_no_margin, ablation_no_plausibility - proposed with one rule switched off

Tiers: reject -> review, and review -> approve.
Reported per method and tier: coverage, cost (proximity), levers changed
(sparsity), plausible and robust shares of the best option, latency.
Also reports recourse coverage and cost by age band and by children, for the
proposed method (burden of recourse across groups).

Run: python3 -m src.recourse_eval [--n 300] [--dice 20]
"""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd

from src.decision import decide
from src.inference import Scorer
from src.recourse import (BAND, DEFAULT_MARGIN, INCOME_SCALES, CREDIT_SCALES, INSTALMENT_SCALES,
                          _Evaluator, _cost, _levers, _logit, _qualifies, apply_changes,
                          default_plausibility, find_recourse)

warnings.filterwarnings("ignore")

RANDOM_BUDGET = 300
LO = np.array([min(CREDIT_SCALES), 1.0, min(INSTALMENT_SCALES)])
HI = np.array([1.0, max(INCOME_SCALES), 1.0])


def summarise_option(sc, opt, plaus, base_band):
    """Puts every method's best option on the same footing: re-scored from scratch."""
    if opt is None:
        return {"found": False}
    out = sc.score(apply_changes(opt["raw"], opt["changes"]))
    p = out["probability"]
    band = BAND[out["decision"]]
    cut = sc.reject_from if base_band == BAND["reject"] else sc.approve_below
    margin = float(_logit(cut) - _logit(p))
    prof = np.log(np.clip([[opt["row"][c] for c in ("AMT_INCOME_TOTAL", "AMT_CREDIT",
                                                    "AMT_ANNUITY", "AMT_GOODS_PRICE")]], 1, None))
    return {
        "found": True,
        "improves": band < base_band,
        "cost": opt["cost"],
        "levers": len(opt["levers"]),
        "robust": margin >= DEFAULT_MARGIN,
        "margin": margin,
        "plausible": bool(plaus.distance(prof)[0] <= plaus.threshold),
    }


def from_engine(sc, raw, res):
    if not res:
        return None
    best = res[0]
    row = dict(sc.build_row(raw))
    for ch in best["changes"]:
        row[ch["feature"]] = ch["recommended"]
    return {"raw": raw, "changes": best["changes"], "row": row,
            "cost": best["cost"], "levers": best["levers"]}


def changes_for(base, s):
    c, i, a = s
    out = []
    if c < 1:
        out.append({"feature": "AMT_CREDIT", "recommended": base["AMT_CREDIT"] * c})
    if i > 1:
        out.append({"feature": "AMT_INCOME_TOTAL", "recommended": base["AMT_INCOME_TOTAL"] * i})
    if c < 1 or a < 1:
        out.append({"feature": "AMT_ANNUITY", "recommended": base["AMT_ANNUITY"] * c * a})
    return out


def random_search(sc, raw, plaus, rng):
    base = sc.build_row(raw)
    ev = _Evaluator(sc, base, DEFAULT_MARGIN, plaus)
    pts = [(1.0, 1.0, 1.0)] + [tuple(LO + rng.random(3) * (HI - LO)) for _ in range(RANDOM_BUDGET)]
    rows, probs, bands, robust, dist = ev(pts)
    ok = _qualifies(robust, dist, bands[0], plaus)
    ok[0] = False
    if not ok.any():
        return None
    k = min(np.where(ok)[0], key=lambda j: _cost(*pts[j]))
    return {"raw": raw, "changes": changes_for(base, pts[k]), "row": rows[k],
            "cost": _cost(*pts[k]), "levers": list(_levers(*pts[k]))}


class _DiceModel:
    """Lever values in, class probabilities out (class 1 = reaches the better band with margin)."""

    def __init__(self, sc, base, target_cut):
        self.sc, self.base = sc, base
        self.target = _logit(target_cut) - DEFAULT_MARGIN

    def predict_proba(self, df):
        rows = []
        for credit, income, annuity in df[["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY"]].to_numpy(float):
            r = dict(self.base)
            r.update(AMT_CREDIT=credit, AMT_INCOME_TOTAL=income, AMT_ANNUITY=annuity)
            rows.append(r)
        from src.feature_engineering_v2 import add_features
        X = add_features(pd.DataFrame(rows)).reindex(columns=self.sc.features).astype(float)
        z = self.target - self.sc.model.predict(X, raw_score=True)
        q = 1 / (1 + np.exp(-20 * z))
        return np.column_stack([1 - q, q])


def dice_search(sc, raw, plaus, base_band, background):
    import dice_ml
    base = sc.build_row(raw)
    cut = sc.reject_from if base_band == BAND["reject"] else sc.approve_below
    cols = ["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY"]
    bounds = {
        "AMT_CREDIT": [base["AMT_CREDIT"] * LO[0], base["AMT_CREDIT"]],
        "AMT_INCOME_TOTAL": [base["AMT_INCOME_TOTAL"], base["AMT_INCOME_TOTAL"] * HI[1]],
        "AMT_ANNUITY": [base["AMT_ANNUITY"] * LO[0] * LO[2], base["AMT_ANNUITY"]],
    }
    model = _DiceModel(sc, base, cut)
    bg = background[cols].copy()
    bg["outcome"] = (model.predict_proba(bg)[:, 1] > 0.5).astype(int)
    if bg["outcome"].nunique() < 2:
        bg.loc[bg.index[0], "outcome"] = 1 - bg["outcome"].iloc[0]
    data = dice_ml.Data(dataframe=bg, continuous_features=cols, outcome_name="outcome")
    dm = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")
    exp = dice_ml.Dice(data, dm, method="random")
    query = pd.DataFrame([{c: base[c] for c in cols}])
    try:
        res = exp.generate_counterfactuals(query, total_CFs=3, desired_class=1,
                                           permitted_range=bounds, random_seed=0)
        cfs = res.cf_examples_list[0].final_cfs_df
    except Exception:
        return None
    if cfs is None or not len(cfs):
        return None
    best = None
    for _, r in cfs.iterrows():
        c = r["AMT_CREDIT"] / base["AMT_CREDIT"]
        i = r["AMT_INCOME_TOTAL"] / base["AMT_INCOME_TOTAL"]
        a = (r["AMT_ANNUITY"] / base["AMT_ANNUITY"]) / c
        s = (min(c, 1.0), max(i, 1.0), min(a, 1.0))
        if best is None or _cost(*s) < _cost(*best):
            best = s
    row = dict(base)
    for ch in changes_for(base, best):
        row[ch["feature"]] = ch["recommended"]
    for d, num, den in (("CREDIT_TO_GOODS_RATIO", "AMT_CREDIT", "AMT_GOODS_PRICE"),
                        ("CREDIT_INCOME_RATIO", "AMT_CREDIT", "AMT_INCOME_TOTAL")):
        if row[den] > 0:
            row[d] = row[num] / row[den]
    return {"raw": raw, "changes": changes_for(base, best), "row": row,
            "cost": _cost(*best), "levers": list(_levers(*best))}


def age_band(days_birth):
    years = -days_birth / 365.25
    return pd.cut(years, [0, 25, 35, 45, 55, 200], right=False,
                  labels=["<25", "25-35", "35-45", "45-55", "55+"]).astype(str)


def aggregate(df):
    f = df[df["found"]]
    return {
        "applicants": int(len(df)),
        "coverage": round(float(df["found"].mean()), 4),
        "improves_share": round(float(f["improves"].mean()), 4) if len(f) else None,
        "mean_cost": round(float(f["cost"].mean()), 4) if len(f) else None,
        "median_cost": round(float(f["cost"].median()), 4) if len(f) else None,
        "mean_levers": round(float(f["levers"].mean()), 3) if len(f) else None,
        "plausible_share": round(float(f["plausible"].mean()), 4) if len(f) else None,
        "robust_share": round(float(f["robust"].mean()), 4) if len(f) else None,
        "latency_ms_mean": round(float(df["latency_ms"].mean()), 1),
        "latency_ms_p95": round(float(df["latency_ms"].quantile(0.95)), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="applicants per tier")
    ap.add_argument("--dice", type=int, default=20, help="applicants per tier for DiCE (0 = skip)")
    args = ap.parse_args()

    sc = Scorer()
    plaus = default_plausibility()
    if plaus is None:
        raise SystemExit("Run python3 -m src.build_recourse_reference first")

    raw_test = pd.read_csv("data/processed/X_test.csv")
    fe = pd.read_csv("data/processed/X_test_fe_v2.csv")
    bands = decide(sc.model.predict(fe[sc.features]), sc.approve_below, sc.reject_from)
    background = raw_test.sample(1000, random_state=0)

    rng = np.random.default_rng(42)
    rows = []
    for tier, band in (("reject_to_review", "reject"), ("review_to_approve", "review")):
        pick = rng.choice(np.where(bands == band)[0], args.n, replace=False)
        for n_done, i in enumerate(pick):
            raw = raw_test.iloc[i].to_dict()
            base_band = BAND[band]
            meta = {"tier": tier, "row": int(i),
                    "age_band": age_band(pd.Series([raw["DAYS_BIRTH"]]))[0],
                    "has_children": "yes" if raw.get("CNT_CHILDREN", 0) > 0 else "no"}

            runs = {
                "previous": lambda: from_engine(sc, raw, find_recourse(sc, raw, margin=0.0,
                                                                       plausibility=None, refine=False)),
                "proposed": lambda: from_engine(sc, raw, find_recourse(sc, raw)),
                "ablation_no_margin": lambda: from_engine(sc, raw, find_recourse(sc, raw, margin=0.0)),
                "ablation_no_plausibility": lambda: from_engine(sc, raw, find_recourse(sc, raw, plausibility=None)),
                "random": lambda: random_search(sc, raw, plaus, rng),
            }
            if n_done < args.dice:
                runs["dice"] = lambda: dice_search(sc, raw, plaus, base_band, background)

            for method, fn in runs.items():
                t = time.perf_counter()
                opt = fn()
                ms = 1000 * (time.perf_counter() - t)
                rows.append({**meta, "method": method, "latency_ms": ms,
                             **summarise_option(sc, opt, plaus, base_band)})
            if (n_done + 1) % 50 == 0:
                print(f"  {tier}: {n_done + 1}/{args.n}")

    df = pd.DataFrame(rows)
    for col in ("improves", "robust", "plausible"):
        df[col] = df[col].astype("boolean")

    summary = {"margin_logodds": DEFAULT_MARGIN, "plausibility_threshold": plaus.threshold,
               "methods": {}}
    for (tier, method), g in df.groupby(["tier", "method"]):
        summary["methods"].setdefault(tier, {})[method] = aggregate(g)

    prop = df[df["method"] == "proposed"]
    burden = {}
    for tier, g in prop.groupby("tier"):
        burden[tier] = {}
        for attr in ("age_band", "has_children"):
            per = {k: aggregate(v) for k, v in g.groupby(attr)}
            cov = [v["coverage"] for v in per.values()]
            cost = [v["mean_cost"] for v in per.values() if v["mean_cost"] is not None]
            burden[tier][attr] = {
                "groups": {k: {"applicants": v["applicants"], "coverage": v["coverage"],
                               "mean_cost": v["mean_cost"]} for k, v in per.items()},
                "coverage_gap": round(max(cov) - min(cov), 4),
                "cost_gap": round(max(cost) - min(cost), 4) if cost else None,
            }
    summary["recourse_burden_by_group"] = burden

    df.to_csv("reports/recourse_evaluation_per_applicant.csv", index=False)
    with open("reports/recourse_evaluation.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["methods"], indent=2))


if __name__ == "__main__":
    main()
