"""
Counterfactual recourse for rejected applicants.

Searches a bounded grid over three levers the applicant controls:
  credit  - requested loan amount (larger down payment); installment falls with it
  income  - verified income, e.g. adding a co-applicant (up to +50%)
  tenure  - longer repayment, expressed as a lower monthly installment (down to 70%)
Every candidate is rebuilt through the FE v2 path, so derived ratios stay
consistent with the changed inputs. Suggestions are model-based, not causal.
"""
import itertools

import pandas as pd

from src.decision import decide
from src.feature_engineering_v2 import add_features

CREDIT_SCALES = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]
INCOME_SCALES = [1.0, 1.1, 1.2, 1.3, 1.5]
INSTALMENT_SCALES = [1.0, 0.9, 0.8, 0.7]

ALLOWED_FEATURES = {"AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY"}
DERIVED = ("CREDIT_TO_GOODS_RATIO", "CREDIT_INCOME_RATIO")
BAND = {"approve": 0, "review": 1, "reject": 2}


def apply_changes(raw, changes):
    """Applies a recourse to a raw applicant dict (derived ratios get recomputed)."""
    out = {k: v for k, v in raw.items() if k not in DERIVED}
    for ch in changes:
        out[ch["feature"]] = ch["recommended"]
    return out


def _candidate(base, c, i, a):
    r = dict(base)
    r["AMT_CREDIT"] = base["AMT_CREDIT"] * c
    r["AMT_INCOME_TOTAL"] = base["AMT_INCOME_TOTAL"] * i
    r["AMT_ANNUITY"] = base["AMT_ANNUITY"] * c * a
    if r["AMT_GOODS_PRICE"] > 0:
        r["CREDIT_TO_GOODS_RATIO"] = r["AMT_CREDIT"] / r["AMT_GOODS_PRICE"]
    if r["AMT_INCOME_TOTAL"] > 0:
        r["CREDIT_INCOME_RATIO"] = r["AMT_CREDIT"] / r["AMT_INCOME_TOTAL"]
    return r


def _change(feature, label, original, recommended):
    return {
        "feature": feature,
        "label": label,
        "original": float(original),
        "recommended": float(recommended),
    }


def _describe(base, scales, row, p_before, p_after, band, cost):
    c, i, a = scales
    changes = []
    if c < 1:
        changes.append(_change("AMT_CREDIT", "Requested loan amount (larger down payment)",
                               base["AMT_CREDIT"], row["AMT_CREDIT"]))
    if i > 1:
        changes.append(_change("AMT_INCOME_TOTAL", "Verified income (e.g. add a co-applicant)",
                               base["AMT_INCOME_TOTAL"], row["AMT_INCOME_TOTAL"]))
    if c < 1 or a < 1:
        changes.append(_change("AMT_ANNUITY", "Monthly installment",
                               base["AMT_ANNUITY"], row["AMT_ANNUITY"]))
    levers = [name for name, on in (("credit", c < 1), ("income", i > 1), ("tenure", a < 1)) if on]
    return {
        "levers": levers,
        "changes": changes,
        "num_changes": len(levers),
        "original_probability": float(p_before),
        "new_probability": float(p_after),
        "new_decision": band.upper(),
        "reaches_approve": band == "approve",
        "success": True,
        "cost": float(cost),
        "feasibility_score": round(1.0 / (1.0 + cost), 3),
    }


def find_recourse(scorer, raw, n=3):
    """
    Returns up to n recourse options that move the applicant to a better band,
    cheapest first, at most one per distinct lever combination.
    Returns [] if the applicant is already in the best band or nothing helps.
    """
    base = scorer.build_row(raw)
    grid = list(itertools.product(CREDIT_SCALES, INCOME_SCALES, INSTALMENT_SCALES))
    rows = [_candidate(base, c, i, a) for c, i, a in grid]

    X = add_features(pd.DataFrame(rows)).reindex(columns=scorer.features).astype(float)
    probs = scorer.model.predict(X)
    bands = decide(probs, scorer.approve_below, scorer.reject_from)

    base_band = BAND[bands[0]]   # grid[0] is (1, 1, 1): the unchanged applicant
    if base_band == 0:
        return []

    best = {}
    for k, (c, i, a) in enumerate(grid):
        if BAND[bands[k]] >= base_band:
            continue
        levers = tuple(name for name, on in (("credit", c < 1), ("income", i > 1), ("tenure", a < 1)) if on)
        cost = (1 - c) + (i - 1) + (1 - a)
        if levers not in best or cost < best[levers][0]:
            best[levers] = (cost, k)

    ranked = sorted(best.values())[:n]
    return [
        _describe(base, grid[k], rows[k], probs[0], probs[k], bands[k], cost)
        for cost, k in ranked
    ]