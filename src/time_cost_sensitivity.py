"""
How much does the price put on waiting change the advice? (KNOWN_ISSUES #73)

Re-runs the recourse search (all four levers) with the time cost per month at
0.02, 0.04 (the value in src/recourse.py) and 0.08, on the same held-out
applicants as src/recourse_eval.py. Qualification does not depend on the time
cost, so coverage is unchanged; what can change is which option is offered
first and which three are shown.

Run: python3 -m src.time_cost_sensitivity [--n 300]
Writes reports/time_cost_sensitivity.json.
"""
import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import src.recourse as recourse
from src.decision import decide
from src.inference import Scorer

warnings.filterwarnings("ignore")
REPORT = Path("reports/time_cost_sensitivity.json")
COSTS = [0.02, 0.04, 0.08]
BASELINE = 0.04


def option_key(opt):
    """What the applicant is told: which levers, how long to wait, and the new amounts."""
    amounts = tuple(round(c["recommended"]) for c in opt["changes"] if c["feature"].startswith("AMT_"))
    return tuple(opt["levers"]), opt["wait_months"], amounts


def compare(runs, baseline):
    """Per time cost: how the first option and the shown set differ from the baseline run."""
    out = {}
    base = runs[baseline]
    for cost, res in runs.items():
        firsts = [r[0] for r in res if r]
        same_first = [bool(r) and bool(b) and option_key(r[0]) == option_key(b[0]) for r, b in zip(res, base) if b]
        same_set = [{option_key(o) for o in r} == {option_key(o) for o in b} for r, b in zip(res, base) if b]
        out[str(cost)] = {
            "coverage": round(float(np.mean([bool(r) for r in res])), 4),
            "first_option_waits": round(float(np.mean([o["wait_months"] > 0 for o in firsts])), 4) if firsts else None,
            "first_option_mean_wait_months": round(float(np.mean([o["wait_months"] for o in firsts])), 2) if firsts else None,
            "first_option_mean_levers": round(float(np.mean([len(o["levers"]) for o in firsts])), 3) if firsts else None,
            "same_first_option_as_baseline": round(float(np.mean(same_first)), 4) if same_first else None,
            "same_options_shown_as_baseline": round(float(np.mean(same_set)), 4) if same_set else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="applicants per tier")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    try:
        os.nice(10)
    except OSError:
        pass

    sc = Scorer()
    raw_test = pd.read_csv("data/processed/X_test.csv")
    fe = pd.read_csv("data/processed/X_test_fe_v2.csv")
    bands = decide(sc.model.predict(fe[sc.features]), sc.approve_below, sc.reject_from)
    pick_rng = np.random.default_rng(42)   # same applicants as src/recourse_eval.py
    picks = {band: pick_rng.choice(np.where(bands == band)[0], args.n, replace=False) for band in ("reject", "review")}

    result = {"applicants_per_tier": args.n, "baseline_cost_per_month": BASELINE, "tiers": {}}
    with threadpool_limits(limits=args.threads):
        for tier, band in (("reject_to_review", "reject"), ("review_to_approve", "review")):
            runs = {}
            for cost in COSTS:
                recourse.TIME_COST_PER_MONTH = cost   # read by recourse._cost at call time
                runs[cost] = [recourse.find_recourse(sc, raw_test.iloc[i].to_dict()) for i in picks[band]]
                print(f"  {tier}: time cost {cost} done", flush=True)
            result["tiers"][tier] = compare(runs, BASELINE)
    recourse.TIME_COST_PER_MONTH = BASELINE

    REPORT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["tiers"], indent=2))


if __name__ == "__main__":
    main()
