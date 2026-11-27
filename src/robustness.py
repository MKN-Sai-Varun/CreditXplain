"""
Does a recourse option still hold after the model is retrained?

The bootstrap models from src/train_bootstrap_models.py stand in for "the bank
retrained the model". An option holds on a model if, with the option applied,
that model puts the applicant in the promised band (Review or better for a
rejected applicant, Approve for one in Review). Two cutoff policies are checked:
- own: each retrained model with the cutoffs its own training run selected
- fixed: every model with the final model's cutoffs
"""
import json
from pathlib import Path

import joblib
import numpy as np

from src.recourse import BAND

REPORT = Path("reports/bootstrap_models.json")


def band_index(prob, approve_below, reject_from):
    if prob < approve_below:
        return BAND["approve"]
    if prob >= reject_from:
        return BAND["reject"]
    return BAND["review"]


def count_holds(probs, cutoffs, promised):
    """
    probs: one probability per model; cutoffs: one (approve_below, reject_from)
    per model; promised: the band index the option promises.
    Returns how many models put the applicant in the promised band or better.
    """
    return int(sum(band_index(p, a, r) <= promised for p, (a, r) in zip(probs, cutoffs)))


class Ensemble:
    def __init__(self, models, own_cutoffs, fixed_cutoffs):
        self.models = models
        self.own = list(own_cutoffs)
        self.fixed = [tuple(fixed_cutoffs)] * len(models)

    def __len__(self):
        return len(self.models)

    @classmethod
    def load(cls, report=REPORT, ks=None):
        """
        The bootstrap models numbered in ks (all of them if None), or None if
        none of those have been trained.
        """
        if not Path(report).exists():
            return None
        r = json.loads(Path(report).read_text())
        members = [m for m in r.get("models", []) if Path(m["path"]).exists() and (ks is None or m["k"] in ks)]
        if not members:
            return None
        final = r["final_model"]
        return cls([joblib.load(m["path"]) for m in members],
                   [(m["approve_threshold"], m["reject_threshold"]) for m in members],
                   (final["approve_threshold"], final["reject_threshold"]))

    def bands(self, X):
        """Band index of every row on every model, with each model's own cutoffs (models x rows)."""
        probs = np.vstack([m.predict(X) for m in self.models])
        approve = np.array([c[0] for c in self.own])[:, None]
        reject = np.array([c[1] for c in self.own])[:, None]
        return np.where(probs < approve, BAND["approve"], np.where(probs >= reject, BAND["reject"], BAND["review"]))

    def probabilities(self, X):
        """One probability per model for a single-row frame."""
        return np.array([float(m.predict(X)[0]) for m in self.models])

    def holds(self, X, promised):
        """(models holding with their own cutoffs, models holding with the final cutoffs)."""
        p = self.probabilities(X)
        return count_holds(p, self.own, promised), count_holds(p, self.fixed, promised)
