"""
Counterfactual recourse for applicants who are not approved.

Searches a bounded space over four levers the applicant controls:
  credit  - requested loan amount (larger down payment); installment falls with it
  income  - verified income, e.g. adding a co-applicant (up to +50%)
  tenure  - longer repayment, expressed as a lower yearly repayment (annuity, down to 70%)
  wait    - re-apply after 3, 6 or 12 months, staying in the same job and making
            no new credit applications meanwhile: job tenure and the age of the
            ID, registration, phone and car records grow, and credit-bureau
            enquiries age out of their windows. The applicant's own age is held
            fixed, so no option relies on getting older.
The applicant can lock levers and cap the loan reduction and the waiting time.

A rejected applicant gets options that reach Review; an applicant in Review
gets options that reach Approve. Optionally (ensemble=...), an option must also
reach the better band on enough retrained models, so it is likely to survive a
retrain of the model. A suggestion is kept only if
  - it reaches the better band even with a safety margin on the cutoff
    (so a small model change does not undo it), and
  - its loan/income profile looks like real applicants the model does not
    reject (kNN distance, see src/build_recourse_reference.py).
The cheapest grid point per lever combination is then pulled back towards the
applicant's own values to find the smallest change that still qualifies.

Every candidate is rebuilt through the FE v2 path, so derived ratios stay
consistent. Suggestions are model-based, not causal.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.decision import decide
from src.feature_engineering_v2 import add_features

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "models" / "recourse_reference.npz"

CREDIT_SCALES = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]
INCOME_SCALES = [1.0, 1.1, 1.2, 1.3, 1.5]
INSTALMENT_SCALES = [1.0, 0.9, 0.8, 0.7]
WAIT_MONTHS = [0, 3, 6, 12]
REFINE_STEPS = 20

# 12 months of waiting costs about as much as halving the loan
TIME_COST_PER_MONTH = 0.04
DAYS_PER_MONTH = 30.44
LEVERS = ("credit", "income", "tenure", "wait")

# log-odds the new score must clear the cutoff by (0.1 is about 10% on the odds)
DEFAULT_MARGIN = 0.1

# days-before-application fields that grow while the applicant waits
TIME_DAYS = ("DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE")
ENQUIRY_SHORT = ("AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK",
                 "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT")
ENQUIRY_YEAR = "AMT_REQ_CREDIT_BUREAU_YEAR"
TIME_FEATURES = set(TIME_DAYS) | set(ENQUIRY_SHORT) | {ENQUIRY_YEAR, "OWN_CAR_AGE", "WAIT_MONTHS"}

ALLOWED_FEATURES = {"AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY"} | TIME_FEATURES
DERIVED = ("CREDIT_TO_GOODS_RATIO", "CREDIT_INCOME_RATIO")
BAND = {"approve": 0, "review": 1, "reject": 2}

PROFILE_COLUMNS = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
K_NEIGHBOURS = 5


def profile_matrix(df):
    return np.log(df[PROFILE_COLUMNS].to_numpy(dtype=float).clip(min=1.0))


class Plausibility:
    def __init__(self, reference, mean, std, threshold):
        self.mean, self.std = np.asarray(mean), np.asarray(std)
        self.threshold = float(threshold)
        self.nn = NearestNeighbors(n_neighbors=K_NEIGHBOURS).fit((reference - self.mean) / self.std)

    def distance(self, profiles):
        d, _ = self.nn.kneighbors((np.asarray(profiles) - self.mean) / self.std)
        return d.mean(axis=1)

    @classmethod
    def load(cls, path=REFERENCE_PATH):
        if not Path(path).exists():
            return None
        z = np.load(path)
        return cls(z["reference"], z["mean"], z["std"], z["threshold"])


_plausibility = None


def default_plausibility():
    global _plausibility
    if _plausibility is None:
        _plausibility = Plausibility.load() or False
    return _plausibility or None


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def _shift(cutoff, margin):
    return float(1 / (1 + np.exp(-(_logit(cutoff) - margin))))


def apply_changes(raw, changes):
    """Applies a recourse to a raw applicant dict (derived ratios get recomputed)."""
    out = {k: v for k, v in raw.items() if k not in DERIVED}
    for ch in changes:
        if ch["feature"] != "WAIT_MONTHS":
            out[ch["feature"]] = ch["recommended"]
    return out


def _present(row, col):
    v = row.get(col)
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def _pass_time(row, months):
    """The applicant's record after `months` with the same job and no new credit applications."""
    r = dict(row)   # always a copy: callers modify the result
    if months <= 0:
        return r
    days = months * DAYS_PER_MONTH
    for col in TIME_DAYS:
        if col == "DAYS_EMPLOYED" and r.get("IS_EMPLOYED_ANOMALY", 0) == 1:
            continue   # not in employment: no tenure to build
        if _present(r, col):
            r[col] = r[col] - days
    if _present(r, "OWN_CAR_AGE") and r.get("FLAG_OWN_CAR_Y", 1) == 1:
        r["OWN_CAR_AGE"] = r["OWN_CAR_AGE"] + months / 12
    # enquiry windows are exclusive (hour, day, week, month, quarter, year):
    # after 3-6 months every recent enquiry sits in the year window; after 12
    # months none is within a year any more
    recent = sum(r[c] for c in ENQUIRY_SHORT if _present(r, c))
    for c in ENQUIRY_SHORT:
        if _present(r, c):
            r[c] = 0.0
    if _present(r, ENQUIRY_YEAR):
        r[ENQUIRY_YEAR] = 0.0 if months >= 12 else r[ENQUIRY_YEAR] + recent
    return r


def _candidate(base, c, i, a, w=0):
    r = _pass_time(base, w)
    r["AMT_CREDIT"] = base["AMT_CREDIT"] * c
    r["AMT_INCOME_TOTAL"] = base["AMT_INCOME_TOTAL"] * i
    r["AMT_ANNUITY"] = base["AMT_ANNUITY"] * c * a
    if r["AMT_GOODS_PRICE"] > 0:
        r["CREDIT_TO_GOODS_RATIO"] = r["AMT_CREDIT"] / r["AMT_GOODS_PRICE"]
    if r["AMT_INCOME_TOTAL"] > 0:
        r["CREDIT_INCOME_RATIO"] = r["AMT_CREDIT"] / r["AMT_INCOME_TOTAL"]
    return r


def _levers(c, i, a, w=0):
    return tuple(name for name, on in (("credit", c < 1), ("income", i > 1),
                                       ("tenure", a < 1), ("wait", w > 0)) if on)


def _cost(c, i, a, w=0):
    return (1 - c) + (i - 1) + (1 - a) + TIME_COST_PER_MONTH * w


class _Evaluator:
    """Scores candidate scale tuples (credit, income, installment[, wait months]) for one applicant."""

    def __init__(self, scorer, base, margin, plausibility, ensemble=None):
        self.scorer, self.base, self.plaus, self.ensemble = scorer, base, plausibility, ensemble
        self.strict = (_shift(scorer.approve_below, margin), _shift(scorer.reject_from, margin))

    def __call__(self, scales):
        rows = [_candidate(self.base, *s) for s in scales]
        frame = pd.DataFrame(rows)
        X = add_features(frame).reindex(columns=self.scorer.features).astype(float)
        probs = self.scorer.model.predict(X)
        bands = np.array([BAND[b] for b in decide(probs, self.scorer.approve_below, self.scorer.reject_from)])
        robust = np.array([BAND[b] for b in decide(probs, *self.strict)])
        dist = self.plaus.distance(profile_matrix(frame)) if self.plaus else np.full(len(rows), np.nan)
        # band of every candidate on every retrained model (models x candidates)
        ens = self.ensemble.bands(X) if self.ensemble is not None else None
        return rows, probs, bands, robust, dist, ens


def _agreeing(ens, base_band):
    """How many retrained models put each candidate in a better band."""
    return (ens < base_band).sum(axis=0)


def _qualifies(robust, dist, base_band, plaus, ens=None, min_agree=None):
    ok = robust < base_band
    if plaus:
        ok &= dist <= plaus.threshold
    if ens is not None:
        ok &= _agreeing(ens, base_band) >= min_agree
    return ok


TIME_LABELS = {
    "DAYS_EMPLOYED": "Days in current job",
    "DAYS_REGISTRATION": "Days since registration change",
    "DAYS_ID_PUBLISH": "Days since ID document issued",
    "DAYS_LAST_PHONE_CHANGE": "Days since phone change",
    "OWN_CAR_AGE": "Car age (years)",
    ENQUIRY_YEAR: "Credit enquiries in the last year",
}


def _describe(base, scales, row, p_before, p_after, band, cost, margin_left, dist, plaus):
    c, i, a, w = (tuple(scales) + (0,))[:4]
    changes = []
    if w > 0:
        changes.append(_change("WAIT_MONTHS", "Re-apply after this many months, staying in the same job "
                               "and making no new credit applications", 0, w, "wait"))
        for col in TIME_DAYS + ENQUIRY_SHORT + (ENQUIRY_YEAR, "OWN_CAR_AGE"):
            if _present(base, col) and _present(row, col) and row[col] != base[col]:
                label = TIME_LABELS.get(col, "Recent credit enquiries (" + col.rsplit("_", 1)[-1].lower() + ")")
                # DAYS_* are stored as negative numbers; show them as positive day counts
                sign = -1 if col.startswith("DAYS_") else 1
                changes.append(_change(col, label, base[col], row[col], "wait", display_sign=sign))
    if c < 1:
        changes.append(_change("AMT_CREDIT", "Requested loan amount (larger down payment)",
                               base["AMT_CREDIT"], row["AMT_CREDIT"], "credit"))
    if i > 1:
        changes.append(_change("AMT_INCOME_TOTAL", "Verified yearly income (e.g. add a co-applicant)",
                               base["AMT_INCOME_TOTAL"], row["AMT_INCOME_TOTAL"], "income"))
    if c < 1 or a < 1:
        changes.append(_change("AMT_ANNUITY", "Yearly loan repayment (annuity)",
                               base["AMT_ANNUITY"], row["AMT_ANNUITY"], "tenure" if a < 1 else "credit"))
    levers = list(_levers(c, i, a, w))
    name = {v: k for k, v in BAND.items()}[band]
    return {
        "levers": levers,
        "changes": changes,
        "num_changes": len(levers),
        "wait_months": int(w),
        "original_probability": float(p_before),
        "new_probability": float(p_after),
        "new_decision": name.upper(),
        "reaches_approve": name == "approve",
        "success": True,
        "cost": float(cost),
        "feasibility_score": round(1.0 / (1.0 + cost), 3),
        "robust_margin": float(margin_left),
        "plausible": None if not plaus else bool(dist <= plaus.threshold),
        "plausibility_distance": None if not plaus else float(dist),
    }


def _change(feature, label, original, recommended, lever, display_sign=1):
    out = {
        "feature": feature,
        "label": label,
        "lever": lever,
        "original": float(original),
        "recommended": float(recommended),
    }
    if display_sign != 1:
        out["display_original"] = float(original) * display_sign
        out["display_recommended"] = float(recommended) * display_sign
    return out


def _dominates(a, b):
    """True if option a asks for no more than option b on every lever (and is a different option)."""
    (ca, ia, aa, wa), (cb, ib, ab, wb) = a, b
    return a != b and ca >= cb and ia <= ib and aa >= ab and wa <= wb


def _search_space(locked, max_loan_reduction, max_wait_months):
    locked = set(locked or ())
    unknown = locked - set(LEVERS)
    if unknown:
        raise ValueError(f"unknown levers: {sorted(unknown)}; choose from {list(LEVERS)}")
    if not 0 <= max_loan_reduction <= 1 - min(CREDIT_SCALES):
        raise ValueError(f"max_loan_reduction must be between 0 and {1 - min(CREDIT_SCALES):.2f}")
    credit = [1.0] if "credit" in locked else [c for c in CREDIT_SCALES if 1 - c <= max_loan_reduction + 1e-9]
    income = [1.0] if "income" in locked else INCOME_SCALES
    tenure = [1.0] if "tenure" in locked else INSTALMENT_SCALES
    wait = [0] if "wait" in locked else [m for m in WAIT_MONTHS if m <= max_wait_months]
    return list(itertools.product(credit, income, tenure, wait))


def find_recourse(scorer, raw, n=3, margin=DEFAULT_MARGIN, plausibility="default", refine=True,
                  locked=(), max_loan_reduction=0.5, max_wait_months=12, ensemble=None, min_agree=None):
    """
    Returns up to n recourse options that move the applicant to a better band,
    cheapest first, at most one per distinct lever combination.
    Returns [] if the applicant is already approved or nothing qualifies.

    plausibility: "default" loads models/recourse_reference.npz if present;
    None switches the check off; or pass a Plausibility object.
    locked: levers the applicant cannot use, from ("credit", "income", "tenure", "wait").
    max_loan_reduction: largest share the loan may shrink by (0 to 0.5).
    max_wait_months: longest the applicant is willing to wait (0, 3, 6 or 12).
    ensemble: retrained models (src.robustness.Ensemble); an option must then reach
    the better band on at least min_agree of them (default: all).
    """
    grid = _search_space(locked, max_loan_reduction, max_wait_months)
    base = scorer.build_row(raw)
    plaus = default_plausibility() if plausibility == "default" else plausibility
    evaluate = _Evaluator(scorer, base, margin, plaus, ensemble)
    if ensemble is not None:
        min_agree = len(ensemble) if min_agree is None else min_agree
        if not 1 <= min_agree <= len(ensemble):
            raise ValueError(f"min_agree must be between 1 and {len(ensemble)}")

    rows, probs, bands, robust, dist, ens = evaluate(grid)   # grid[0] is the unchanged applicant
    base_band = bands[0]
    if base_band == BAND["approve"]:
        return []

    ok = _qualifies(robust, dist, base_band, plaus, ens, min_agree)
    agree = _agreeing(ens, base_band) if ens is not None else np.full(len(grid), -1)
    found = {}   # lever combination -> (scales, row, prob, band, distance) of its cheapest option
    for k in np.where(ok)[0]:
        s = grid[k]
        lv = _levers(*s)
        if lv and (lv not in found or _cost(*s) < _cost(*found[lv][0])):
            found[lv] = (s, rows[k], probs[k], bands[k], dist[k], agree[k])

    if refine and found:
        # walk each winner back towards the applicant's own values
        # (the waiting time stays as found; only the loan levers are pulled back)
        ts = np.linspace(1 / REFINE_STEPS, 1, REFINE_STEPS)
        ray = [(lv, tuple(1 - t * (1 - v) for v in f[0][:3]) + (f[0][3],))
               for lv, f in found.items() for t in ts]
        r_rows, r_probs, r_bands, r_robust, r_dist, r_ens = evaluate([s for _, s in ray])
        r_ok = _qualifies(r_robust, r_dist, base_band, plaus, r_ens, min_agree)
        r_agree = _agreeing(r_ens, base_band) if r_ens is not None else np.full(len(ray), -1)
        for k in range(len(ray)):
            lv, s = ray[k]
            if r_ok[k] and _cost(*s) < _cost(*found[lv][0]):
                found[lv] = (s, r_rows[k], r_probs[k], r_bands[k], r_dist[k], r_agree[k])

    cut = {BAND["review"]: scorer.reject_from, BAND["approve"]: scorer.approve_below}
    ranked = [v for v in sorted(found.values(), key=lambda v: _cost(*v[0]))
              if not any(_dominates(u[0], v[0]) for u in found.values() if u is not v)][:n]
    options = []
    for s, row, p, b, d, a in ranked:
        opt = _describe(base, s, row, probs[0], p, b, _cost(*s), float(_logit(cut[b]) - _logit(p)), d, plaus)
        if ensemble is not None:
            opt["ensemble_agreement"] = f"{int(a)}/{len(ensemble)}"
        options.append(opt)
    return options


def explain_change(scorer, shap_explainer, raw, option, top=3, only_changed=True):
    """
    Which model features moved the score most when the option is applied (SHAP, log-odds).
    only_changed keeps features whose value the option changes: SHAP can shift credit onto
    unchanged features through interactions, and naming those (e.g. age) would mislead.
    """
    X = pd.concat([scorer.frame(raw), scorer.frame(apply_changes(raw, option["changes"]))])
    sv = shap_explainer.shap_values(X)
    sv = np.asarray(sv[-1] if isinstance(sv, list) else sv)
    delta = sv[1] - sv[0]
    if only_changed:
        moved = ~np.isclose(X.iloc[0].to_numpy(float), X.iloc[1].to_numpy(float), equal_nan=True)
        delta = np.where(moved, delta, 0.0)
    order = [j for j in np.argsort(-np.abs(delta)) if delta[j] != 0][:top]
    return [{"feature": X.columns[j], "delta": float(delta[j])} for j in order]
