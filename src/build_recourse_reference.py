"""
Builds the reference set the recourse search uses to reject implausible
suggestions: the loan/income profile of training applicants the model does
not reject.

A suggestion is plausible when its distance to the nearest real profiles is
no larger than what real non-rejected applicants show among themselves
(95th percentile, measured on a held-back calibration split).

Run: python3 -m src.build_recourse_reference
Writes models/recourse_reference.npz (gitignored, like the model).
"""
import json

import joblib
import numpy as np
import pandas as pd

from src.decision import decide
from src.recourse import PROFILE_COLUMNS, profile_matrix, Plausibility

REFERENCE_SIZE = 20000
CALIBRATION_SIZE = 5000
PERCENTILE = 95


def main():
    card = json.load(open("models/final_model_card.json"))
    model = joblib.load("models/final_lgbm.pkl")
    X = pd.read_csv("data/processed/X_train_fe_v2.csv")

    bands = decide(model.predict(X[card["features_used"]]),
                   card["approve_threshold"], card["reject_threshold"])
    ok = X.loc[bands != "reject", PROFILE_COLUMNS]
    ok = ok[(ok > 0).all(axis=1)]

    rng = np.random.default_rng(42)
    pick = rng.permutation(len(ok))[: REFERENCE_SIZE + CALIBRATION_SIZE]
    ref = profile_matrix(ok.iloc[pick[:REFERENCE_SIZE]])
    cal = profile_matrix(ok.iloc[pick[REFERENCE_SIZE:]])

    mean, std = ref.mean(axis=0), ref.std(axis=0)
    p = Plausibility(ref, mean, std, threshold=np.inf)
    threshold = float(np.percentile(p.distance(cal), PERCENTILE))

    np.savez("models/recourse_reference.npz", reference=ref, mean=mean, std=std,
             threshold=threshold, columns=np.array(PROFILE_COLUMNS))
    print(f"non-rejected training applicants: {len(ok):,}")
    print(f"reference {len(ref):,}, calibration {len(cal):,}, "
          f"threshold (p{PERCENTILE} kNN distance) = {threshold:.4f}")


if __name__ == "__main__":
    main()
