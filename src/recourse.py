"""
Counterfactual recourse for applicants who are not approved.

Searches a bounded space over three levers the applicant controls:
  credit  - requested loan amount (larger down payment); installment falls with it
  income  - verified income, e.g. adding a co-applicant (up to +50%)
  tenure  - longer repayment, expressed as a lower monthly installment (down to 70%)

A rejected applicant gets options that reach Review; an applicant in Review
gets options that reach Approve. A suggestion is kept only if
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
REFINE_STEPS = 20

# log-odds the new score must clear the cutoff by (0.1 is about 10% on the odds)
DEFAULT_MARGIN = 0.1

ALLOWED_FEATURES = {"AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY"}
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


def _levers(c, i, a):
    return tuple(name for name, on in (("credit", c < 1), ("income", i > 1), ("tenure", a < 1)) if on)


def _cost(c, i, a):
    return (1 - c) + (i - 1) + (1 - a)


class _Evaluator:
    """Scores candidate scale triples for one applicant."""

    def __init__(self, scorer, base, margin, plausibility):
        self.scorer, self.base, self.plaus = scorer, base, plausibility
        self.strict = (_shift(scorer.approve_below, margin), _shift(scorer.reject_from, margin))

    def __call__(self, scales):
        rows = [_candidate(self.base, *s) for s in scales]
        frame = pd.DataFrame(rows)
        X = add_features(frame).reindex(columns=self.scorer.features).astype(float)
        probs = self.scorer.model.predict(X)
        bands = np.array([BAND[b] for b in decide(probs, self.scorer.approve_below, self.scorer.reject_from)])
        robust = np.array([BAND[b] for b in decide(probs, *self.strict)])
        dist = self.plaus.distance(profile_matrix(frame)) if self.plaus else np.full(len(rows), np.nan)
        return rows, probs, bands, robust, dist


def _qualifies(robust, dist, base_band, plaus):
    ok = robust < base_band
    if plaus:
        ok &= dist <= plaus.threshold
    return ok


def _describe(base, scales, row, p_before, p_after, band, cost, margin_left, dist, plaus):
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
    levers = list(_levers(c, i, a))
    name = {v: k for k, v in BAND.items()}[band]
    return {
        "levers": levers,
        "changes": changes,
        "num_changes": len(levers),
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


def _change(feature, label, original, recommended):
    return {
        "feature": feature,
        "label": label,
        "original": float(original),
        "recommended": float(recommended),
    }


def find_recourse(scorer, raw, n=3, margin=DEFAULT_MARGIN, plausibility="default", refine=True):
    """
    Returns up to n recourse options that move the applicant to a better band,
    cheapest first, at most one per distinct lever combination.
    Returns [] if the applicant is already approved or nothing qualifies.

    plausibility: "default" loads models/recourse_reference.npz if present;
    None switches the check off; or pass a Plausibility object.
    """
    base = scorer.build_row(raw)
    plaus = default_plausibility() if plausibility == "default" else plausibility
    evaluate = _Evaluator(scorer, base, margin, plaus)

    grid = list(itertools.product(CREDIT_SCALES, INCOME_SCALES, INSTALMENT_SCALES))
    rows, probs, bands, robust, dist = evaluate(grid)   # grid[0] is the unchanged applicant
    base_band = bands[0]
    if base_band == BAND["approve"]:
        return []

    ok = _qualifies(robust, dist, base_band, plaus)
    found = {}   # lever combination -> (scales, row, prob, band, distance) of its cheapest option
    for k in np.where(ok)[0]:
        s = grid[k]
        lv = _levers(*s)
        if lv and (lv not in found or _cost(*s) < _cost(*found[lv][0])):
            found[lv] = (s, rows[k], probs[k], bands[k], dist[k])

    if refine and found:
        # walk each winner back towards the applicant's own values
        ts = np.linspace(1 / REFINE_STEPS, 1, REFINE_STEPS)
        ray = [(lv, tuple(1 - t * (1 - v) for v in f[0])) for lv, f in found.items() for t in ts]
        r_rows, r_probs, r_bands, r_robust, r_dist = evaluate([s for _, s in ray])
        r_ok = _qualifies(r_robust, r_dist, base_band, plaus)
        for k in range(len(ray)):
            lv, s = ray[k]
            if r_ok[k] and _cost(*s) < _cost(*found[lv][0]):
                found[lv] = (s, r_rows[k], r_probs[k], r_bands[k], r_dist[k])

    cut = {BAND["review"]: scorer.reject_from, BAND["approve"]: scorer.approve_below}
    ranked = sorted(found.values(), key=lambda v: _cost(*v[0]))[:n]
    return [
        _describe(base, s, row, probs[0], p, b, _cost(*s),
                  float(_logit(cut[b]) - _logit(p)), d, plaus)
        for s, row, p, b, d in ranked
    ]


def explain_change(scorer, shap_explainer, raw, option, top=3):
    """Which model features moved the score most when the option is applied (SHAP, log-odds)."""
    X = pd.concat([scorer.frame(raw), scorer.frame(apply_changes(raw, option["changes"]))])
    sv = shap_explainer.shap_values(X)
    sv = np.asarray(sv[-1] if isinstance(sv, list) else sv)
    delta = sv[1] - sv[0]
    order = np.argsort(-np.abs(delta))[:top]
    return [{"feature": X.columns[j], "delta": float(delta[j])} for j in order]
