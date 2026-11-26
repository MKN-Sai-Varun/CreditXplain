"""
Local LIME for the final LightGBM model.

LIME on all 300 features does not give a usable local surrogate: rare one-hot
columns pick up large, noisy weights and the fit explains little of the model.
Here LIME perturbs only the K features with the highest LightGBM gain; every
other feature stays at the applicant's own value. Perturbations are drawn from
quartile bins (LIME's default), and the surrogate is fitted to the model's raw
log-odds, the same scale SHAP's TreeExplainer uses.
"""
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.utils import check_random_state

DEFAULT_K = 25
DEFAULT_SAMPLES = 5000


def top_gain_features(model, features, k=DEFAULT_K):
    gain = pd.Series(model.feature_importance("gain"), index=model.feature_name())
    return list(gain.reindex(features).fillna(0).sort_values(ascending=False).index[:k])


class LocalLime:
    def __init__(self, model, features, background, k=DEFAULT_K, num_samples=DEFAULT_SAMPLES):
        self.model = model
        self.features = list(features)
        self.subset = top_gain_features(model, self.features, k)
        self.idx = [self.features.index(f) for f in self.subset]
        self.num_samples = num_samples

        bg = background[self.subset].astype(float)
        self.categorical = [j for j, f in enumerate(self.subset) if bg[f].nunique() <= 2]
        self._bg = bg.values
        self._explainers = {}

    def _explainer(self, seed):
        if seed not in self._explainers:
            self._explainers[seed] = LimeTabularExplainer(
                self._bg,
                feature_names=self.subset,
                categorical_features=self.categorical,
                mode="regression",
                discretize_continuous=True,
                random_state=seed,
            )
        return self._explainers[seed]

    def explain(self, x, num_features=10, seed=0):
        """
        x: one applicant as a 1-D array in self.features order.
        Returns (rows, weights, score): rows sorted by |weight| with feature,
        rule and weight; weights is a full vector over self.subset; score is
        the surrogate's R^2 on the perturbed sample.
        """
        x = np.asarray(x, dtype=float)

        def predict(Z):
            X = np.tile(x, (len(Z), 1))
            X[:, self.idx] = Z
            return self.model.predict(X, raw_score=True)

        explainer = self._explainer(seed)
        # same seed, same answer: reset both generators LIME draws from
        explainer.random_state = check_random_state(seed)
        explainer.discretizer.random_state = check_random_state(seed)
        exp = explainer.explain_instance(
            x[self.idx], predict,
            num_features=len(self.subset),
            num_samples=self.num_samples,
        )
        # in regression mode LIME keeps the real weights under label 1
        # (label 0 holds the same weights negated)
        weights = np.zeros(len(self.subset))
        for j, w in exp.as_map()[1]:
            weights[j] = w

        self.last_exp = exp   # kept so callers can save LIME's own HTML/plot
        names = exp.domain_mapper.discretized_feature_names or self.subset
        rows = [
            {"feature": self.subset[j], "rule": names[j], "weight": float(weights[j])}
            for j in np.argsort(-np.abs(weights))[:num_features]
        ]
        return rows, weights, float(exp.score)
