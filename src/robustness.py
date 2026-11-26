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
    def load(cls, report=REPORT):
        """None if the bootstrap models have not been trained."""
        if not Path(report).exists():
            return None
        r = json.loads(Path(report).read_text())
        members = [m for m in r.get("models", []) if Path(m["path"]).exists()]
        if not members:
            return None
        final = r["final_model"]
        return cls([joblib.load(m["path"]) for m in members],
                   [(m["approve_threshold"], m["reject_threshold"]) for m in members],
                   (final["approve_threshold"], final["reject_threshold"]))

    def probabilities(self, X):
        """One probability per model for a single-row frame."""
        return np.array([float(m.predict(X)[0]) for m in self.models])

    def holds(self, X, promised):
        """(models holding with their own cutoffs, models holding with the final cutoffs)."""
        p = self.probabilities(X)
        return count_holds(p, self.own, promised), count_holds(p, self.fixed, promised)
