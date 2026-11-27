"""
Evaluates recourse on held-out applicants who are not approved.

Methods (all but proposed_with_wait limited to the same three loan levers and bounds):
  previous  - the pre-4.7 engine: grid only, no margin, no plausibility check
  proposed  - src/recourse.py with margin, plausibility and refinement, loan levers only
  proposed_with_wait - proposed plus the waiting lever (3, 6 or 12 months)
  random    - random search over the same box with a similar evaluation budget,
              same margin and plausibility rules
  dice      - DiCE (random method) on the same levers, scored through the
              same FE v2 path, with DiCE's binary post-hoc sparsity search;
              run on a subset (--dice N)
  ablation_no_margin, ablation_no_plausibility - proposed with one rule switched off
  proposed_margin_0p2 - proposed with a 0.2 log-odds margin (margin sensitivity)
  proposed_ensemble, proposed_ensemble_4of5 - proposed (loan levers) whose options
              must also reach the better band on all 5 / at least 4 of the
              search models (bootstrap models 1-5)
  proposed_ensemble_with_wait - proposed_ensemble plus the waiting lever

Bootstrap models are split in two: 1-5 are used by the ensemble methods during
the search, 6-10 only for evaluation, so robustness is never measured on a model
the search has seen. With --robustness, every best option is re-scored on the
evaluation models (see src/robustness.py), and the unchanged applicant is scored
too, as a reference for how much retraining alone moves bands. If models 6-10
have not been trained, the check falls back to models 1-5, which is only fair
for the methods that do not use them.

Tiers: reject -> review, and review -> approve.
Reported per method and tier: coverage, cost (proximity), levers changed
(sparsity), plausible and robust shares of the best option, latency.
Also reports recourse coverage and cost by age band and by children, for the
proposed method (burden of recourse across groups).

Run: python3 -m src.recourse_eval [--n 300] [--dice 20] [--methods dice] [--robustness]

--methods runs only the listed methods and merges their rows into the existing
per-applicant file, so DiCE can be added without re-running the rest.
Rows are saved every 10 applicants and on Ctrl+C. The run uses 2 CPU threads
(--threads) and low priority, so the machine stays responsive.
"""
import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.decision import decide
from src.inference import Scorer
from src.robustness import Ensemble
from src.recourse import (BAND, DEFAULT_MARGIN, INCOME_SCALES, CREDIT_SCALES, INSTALMENT_SCALES,
                          _Evaluator, _cost, _levers, _logit, _qualifies, apply_changes,
                          default_plausibility, find_recourse)

warnings.filterwarnings("ignore")

RANDOM_BUDGET = 300
DICE_BACKGROUND = 200
METHODS = ["previous", "proposed", "proposed_with_wait", "ablation_no_margin", "ablation_no_plausibility",
           "proposed_margin_0p2", "proposed_ensemble", "proposed_ensemble_4of5", "proposed_ensemble_with_wait",
           "random", "dice"]
ENSEMBLE_METHODS = {"proposed_ensemble", "proposed_ensemble_4of5", "proposed_ensemble_with_wait"}
SEARCH_MODELS = range(1, 6)
EVAL_MODELS = range(6, 11)
LOAN_ONLY = ("wait",)
PER_APPLICANT = Path("reports/recourse_evaluation_per_applicant.csv")
SUMMARY = Path("reports/recourse_evaluation.json")
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
        "wait_months": int(opt.get("wait_months", 0)),
    }


def from_engine(sc, raw, res):
    if not res:
        return None
    best = res[0]
    row = dict(sc.build_row(raw))
    for ch in best["changes"]:
        row[ch["feature"]] = ch["recommended"]
    return {"raw": raw, "changes": best["changes"], "row": row,
            "cost": best["cost"], "levers": best["levers"], "wait_months": best.get("wait_months", 0)}


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
    rows, probs, bands, robust, dist, _ = ev(pts)
    ok = _qualifies(robust, dist, bands[0], plaus)
    ok[0] = False
    if not ok.any():
        return None
    k = min(np.where(ok)[0], key=lambda j: _cost(*pts[j]))
    return {"raw": raw, "changes": changes_for(base, pts[k]), "row": rows[k],
            "cost": _cost(*pts[k]), "levers": list(_levers(*pts[k]))}


class _DiceModel:
    """
    Lever values in, class probabilities out (class 1 = reaches the better band
    with margin). Rebuilds the derived ratios like src/recourse.py does, and
    caches single-row calls, which DiCE's sparsity search repeats.
    """
    COLS = ["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY"]

    def __init__(self, sc, base, target_cut):
        from src.feature_engineering_v2 import add_features
        self.sc, self.add_features = sc, add_features
        self.base = pd.DataFrame([base])
        self.target = _logit(target_cut) - DEFAULT_MARGIN
        self.cache = {}

    def _scores(self, levers):
        X = self.base.loc[np.zeros(len(levers), dtype=int)].reset_index(drop=True)
        X[self.COLS] = levers
        if X.at[0, "AMT_GOODS_PRICE"] > 0:
            X["CREDIT_TO_GOODS_RATIO"] = X["AMT_CREDIT"] / X["AMT_GOODS_PRICE"]
        X["CREDIT_INCOME_RATIO"] = X["AMT_CREDIT"] / X["AMT_INCOME_TOTAL"].where(X["AMT_INCOME_TOTAL"] > 0)
        X = self.add_features(X).reindex(columns=self.sc.features).astype(float)
        z = self.target - self.sc.model.predict(X, raw_score=True)
        return 1 / (1 + np.exp(-20 * z))

    def predict_proba(self, df):
        levers = df[self.COLS].to_numpy(float)
        if len(levers) == 1:
            key = tuple(np.round(levers[0], 2))
            if key not in self.cache:
                self.cache[key] = self._scores(levers)[0]
            q = np.array([self.cache[key]])
        else:
            q = self._scores(levers)
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
                                           permitted_range=bounds, random_seed=0,
                                           posthoc_sparsity_algorithm="binary")
        cfs = res.cf_examples_list[0].final_cfs_df
    except Exception:
        return None
    if cfs is None or not len(cfs):
        return None
    # DiCE varies amount and installment independently; our levers tie them
    # (installment = amount x tenure factor). Suggestions outside that space
    # are counted, not clipped, so DiCE is scored on what it actually proposed.
    best, outside, eps = None, 0, 1e-6
    for _, r in cfs.iterrows():
        c = r["AMT_CREDIT"] / base["AMT_CREDIT"]
        i = r["AMT_INCOME_TOTAL"] / base["AMT_INCOME_TOTAL"]
        a = (r["AMT_ANNUITY"] / base["AMT_ANNUITY"]) / c
        inside = (LO[0] - eps <= c <= 1 + eps and 1 - eps <= i <= HI[1] + eps
                  and LO[2] - eps <= a <= 1 + eps)
        if not inside:
            outside += 1
            continue
        s = (min(c, 1.0), max(i, 1.0), min(a, 1.0))
        if best is None or _cost(*s) < _cost(*best):
            best = s
    if best is None:
        return {"outside": outside}
    row = dict(base)
    for ch in changes_for(base, best):
        row[ch["feature"]] = ch["recommended"]
    for d, num, den in (("CREDIT_TO_GOODS_RATIO", "AMT_CREDIT", "AMT_GOODS_PRICE"),
                        ("CREDIT_INCOME_RATIO", "AMT_CREDIT", "AMT_INCOME_TOTAL")):
        if row[den] > 0:
            row[d] = row[num] / row[den]
    return {"raw": raw, "changes": changes_for(base, best), "row": row,
            "cost": _cost(*best), "levers": list(_levers(*best)), "outside": outside}


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
        **({"uses_wait_share": round(float((f["wait_months"] > 0).mean()), 4),
            "mean_wait_months": round(float(f["wait_months"].mean()), 2)}
           if len(f) and "wait_months" in f and f["wait_months"].fillna(0).gt(0).any() else {}),
        "plausible_share": round(float(f["plausible"].mean()), 4) if len(f) else None,
        "robust_share": round(float(f["robust"].mean()), 4) if len(f) else None,
        "latency_ms_mean": round(float(df["latency_ms"].mean()), 1),
        "latency_ms_p95": round(float(df["latency_ms"].quantile(0.95)), 1),
        **({"suggestions_outside_lever_space": int(df["outside"].fillna(0).sum())}
           if "outside" in df and df["outside"].notna().any() else {}),
        **(_robustness(df, f) if "holds_own" in df and df["holds_own"].notna().any() else {}),
    }


def _robustness(df, f):
    """Shares of found options that still hold on every retrained model."""
    n = int(df["n_models"].max())
    held = f.dropna(subset=["holds_own"])
    return {
        "retrained_models": n,
        "holds_on_all_retrained": round(float((held["holds_own"] == n).mean()), 4) if len(held) else None,
        "holds_on_all_retrained_fixed_cutoffs": round(float((held["holds_fixed"] == n).mean()), 4) if len(held) else None,
        "mean_retrained_models_held": round(float(held["holds_own"].mean()), 3) if len(held) else None,
        "usable_and_holds_on_all": round(float((df["found"] & (df["plausible"] == True) & (df["robust"] == True)
                                                & (df["holds_own"] == n)).mean()), 4),
        "unchanged_applicant_already_promised_on_any": round(float((df["base_holds_own"] > 0).mean()), 4),
    }


def summarise(df, plaus):
    df = df.copy()
    for col in ("improves", "robust", "plausible"):
        df[col] = df[col].astype("boolean")

    summary = {"margin_logodds": DEFAULT_MARGIN, "plausibility_threshold": plaus.threshold,
               "methods": {}}
    for (tier, method), g in df.groupby(["tier", "method"]):
        summary["methods"].setdefault(tier, {})[method] = aggregate(g)

    burden = {}
    for method in ("proposed", "proposed_with_wait"):
        for tier, g in df[df["method"] == method].groupby("tier"):
            out = burden.setdefault(method, {}).setdefault(tier, {})
            for attr in ("age_band", "has_children"):
                per = {k: aggregate(v) for k, v in g.groupby(attr)}
                cov = [v["coverage"] for v in per.values()]
                cost = [v["mean_cost"] for v in per.values() if v["mean_cost"] is not None]
                out[attr] = {
                    "groups": {k: {"applicants": v["applicants"], "coverage": v["coverage"],
                                   "mean_cost": v["mean_cost"]} for k, v in per.items()},
                    "coverage_gap": round(max(cov) - min(cov), 4),
                    "cost_gap": round(max(cost) - min(cost), 4) if cost else None,
                }
    summary["recourse_burden_by_group"] = burden
    return summary


def save(rows, methods, plaus):
    """Writes this run's rows, keeping rows of methods that were not run this time."""
    new = pd.DataFrame(rows)
    if PER_APPLICANT.exists():
        old = pd.read_csv(PER_APPLICANT)
        new = pd.concat([old[~old["method"].isin(methods)], new], ignore_index=True)
    new.to_csv(PER_APPLICANT, index=False)
    summary = summarise(new, plaus)
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="applicants per tier")
    ap.add_argument("--dice", type=int, default=20, help="applicants per tier for DiCE")
    ap.add_argument("--methods", default=",".join(METHODS),
                    help=f"comma-separated subset of {METHODS}")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads for the model")
    ap.add_argument("--robustness", action="store_true",
                    help="also check options on the bootstrap models (python3 -m src.train_bootstrap_models)")
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise SystemExit(f"unknown methods: {sorted(unknown)}")

    try:
        os.nice(10)   # keep the machine responsive
    except OSError:
        pass

    sc = Scorer()
    search_models = None
    if ENSEMBLE_METHODS & set(methods):
        search_models = Ensemble.load(ks=SEARCH_MODELS)
        if search_models is None or len(search_models) < len(SEARCH_MODELS):
            raise SystemExit("The ensemble methods need bootstrap models 1-5: python3 -m src.train_bootstrap_models --n 10")
    ensemble = None   # models used to measure robustness
    if args.robustness:
        ensemble = Ensemble.load(ks=EVAL_MODELS)
        if ensemble is None:
            if search_models is not None:
                raise SystemExit("Evaluation models 6-10 are missing: python3 -m src.train_bootstrap_models --n 10")
            ensemble = Ensemble.load(ks=SEARCH_MODELS)
            if ensemble is None:
                raise SystemExit("Run python3 -m src.train_bootstrap_models first")
            print("note: evaluating robustness on models 1-5 (models 6-10 not trained)")
        print(f"robustness check on {len(ensemble)} held-out retrained models")
    plaus = default_plausibility()
    if plaus is None:
        raise SystemExit("Run python3 -m src.build_recourse_reference first")

    raw_test = pd.read_csv("data/processed/X_test.csv")
    fe = pd.read_csv("data/processed/X_test_fe_v2.csv")
    bands = decide(sc.model.predict(fe[sc.features]), sc.approve_below, sc.reject_from)
    background = raw_test.sample(DICE_BACKGROUND, random_state=0)

    # separate generators: the applicants picked never depend on which methods run
    pick_rng = np.random.default_rng(42)
    search_rng = np.random.default_rng(7)
    picks = {band: pick_rng.choice(np.where(bands == band)[0], args.n, replace=False)
             for band in ("reject", "review")}

    rows = []
    limit = threadpool_limits(limits=args.threads)
    try:
        for tier, band in (("reject_to_review", "reject"), ("review_to_approve", "review")):
            for n_done, i in enumerate(picks[band]):
                raw = raw_test.iloc[i].to_dict()
                base_band = BAND[band]
                meta = {"tier": tier, "row": int(i),
                        "age_band": age_band(pd.Series([raw["DAYS_BIRTH"]]))[0],
                        "has_children": "yes" if raw.get("CNT_CHILDREN", 0) > 0 else "no"}
                promised = base_band - 1   # Review for a rejected applicant, Approve for one in Review
                if ensemble is not None:
                    meta["n_models"] = len(ensemble)
                    meta["base_holds_own"], meta["base_holds_fixed"] = ensemble.holds(sc.frame(raw), promised)

                runs = {
                    "previous": lambda: from_engine(sc, raw, find_recourse(sc, raw, margin=0.0, plausibility=None,
                                                                           refine=False, locked=LOAN_ONLY)),
                    "proposed": lambda: from_engine(sc, raw, find_recourse(sc, raw, locked=LOAN_ONLY)),
                    "proposed_with_wait": lambda: from_engine(sc, raw, find_recourse(sc, raw)),
                    "ablation_no_margin": lambda: from_engine(sc, raw, find_recourse(sc, raw, margin=0.0,
                                                                                     locked=LOAN_ONLY)),
                    "ablation_no_plausibility": lambda: from_engine(sc, raw, find_recourse(sc, raw, plausibility=None,
                                                                                           locked=LOAN_ONLY)),
                    "proposed_margin_0p2": lambda: from_engine(sc, raw, find_recourse(sc, raw, margin=0.2,
                                                                                      locked=LOAN_ONLY)),
                    "proposed_ensemble": lambda: from_engine(sc, raw, find_recourse(
                        sc, raw, locked=LOAN_ONLY, ensemble=search_models)),
                    "proposed_ensemble_4of5": lambda: from_engine(sc, raw, find_recourse(
                        sc, raw, locked=LOAN_ONLY, ensemble=search_models, min_agree=4)),
                    "proposed_ensemble_with_wait": lambda: from_engine(sc, raw, find_recourse(
                        sc, raw, ensemble=search_models)),
                    "random": lambda: random_search(sc, raw, plaus, search_rng),
                    "dice": lambda: dice_search(sc, raw, plaus, base_band, background),
                }
                for method in methods:
                    if method == "dice" and n_done >= args.dice:
                        continue
                    t = time.perf_counter()
                    opt = runs[method]()
                    ms = 1000 * (time.perf_counter() - t)
                    extra = {}
                    if opt is not None and "outside" in opt:
                        extra["outside"] = opt.pop("outside")
                        opt = opt if "raw" in opt else None
                    if ensemble is not None and opt is not None:
                        X = sc.frame(apply_changes(opt["raw"], opt["changes"]))
                        extra["holds_own"], extra["holds_fixed"] = ensemble.holds(X, promised)
                    rows.append({**meta, "method": method, "latency_ms": ms, **extra,
                                 **summarise_option(sc, opt, plaus, base_band)})

                done = n_done + 1
                if done % 10 == 0:
                    save(rows, methods, plaus)
                if done % 50 == 0 or ("dice" in methods and done <= args.dice):
                    print(f"  {tier}: {done}/{args.n}", flush=True)
                if methods == ["dice"] and done >= args.dice:
                    break
    except KeyboardInterrupt:
        print("stopped early; saving what has finished")
    finally:
        limit.restore_original_limits()

    summary = save(rows, methods, plaus)
    print(json.dumps(summary["methods"], indent=2))


if __name__ == "__main__":
    main()
