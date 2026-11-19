import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REQUIRED = ["models/final_lgbm.pkl", "models/final_model_card.json",
            "models/feature_defaults.json", "data/processed/X_test.csv",
            "data/processed/X_test_fe_v2.csv"]

pytestmark = pytest.mark.skipif(
    not all(Path(p).exists() for p in REQUIRED), reason="model/data missing"
)


def test_rejected_application_returns_valid_recourse(client, applicant_headers):
    from backend.services.prediction_service import get_prediction_service
    from src.decision import decide
    from src.recourse import find_recourse

    sc = get_prediction_service().scorer
    base = pd.read_csv("data/processed/X_test.csv").head(600)
    fe = pd.read_csv("data/processed/X_test_fe_v2.csv").head(600)
    dec = decide(sc.model.predict(fe[sc.features]), sc.approve_below, sc.reject_from)

    chosen = None
    for i in np.where(dec == "reject")[0][:40]:
        raw = json.loads(base.iloc[i].to_json())
        if find_recourse(sc, raw):
            chosen = raw
            break
    assert chosen is not None

    app_id = client.post("/api/applications/", json={"features": chosen},
                         headers=applicant_headers).json()["id"]
    p = client.post(f"/api/predictions/{app_id}", headers=applicant_headers)
    assert p.status_code == 200, p.text
    assert p.json()["decision"] == "REJECT"

    d = client.get(f"/api/applications/{app_id}", headers=applicant_headers)
    assert d.status_code == 200, d.text
    cfs = d.json()["explanation"]["counterfactuals"]
    assert cfs and len(cfs) <= 3
    for cf in cfs:
        assert cf["new_decision"] in ("REVIEW", "APPROVE")
        assert cf["changes"]