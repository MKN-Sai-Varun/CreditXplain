"""
Measures how long the system takes, for the paper's latency figures.

Two views over the same held-out applicants:
  api     - real HTTP requests through the FastAPI app (test client, fresh
            in-memory database): submit, decide (score + SHAP + LIME + recourse
            + loan terms), read the decision, applicant PDF, "What if...?",
            compliance PDF
  stages  - the same work called stage by stage, to show where the time goes:
            scoring, SHAP, LIME, recourse (hybrid), applicant PDF

Loading the models at start-up is timed separately (cold start), and one
warm-up applicant is excluded, so per-request figures are steady-state.

Run: python3 -m src.system_latency [--n 100]
Writes reports/system_latency.json.
"""
import argparse
import json
import os
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPORT = Path("reports/system_latency.json")


def summarize(times_ms):
    """mean / median / p95 / max of a list of milliseconds (None when empty)."""
    t = np.asarray([x for x in times_ms if x is not None], dtype=float)
    if not len(t):
        return None
    return {"n": int(len(t)), "mean_ms": round(float(t.mean()), 1), "median_ms": round(float(np.median(t)), 1),
            "p95_ms": round(float(np.percentile(t, 95)), 1), "max_ms": round(float(t.max()), 1)}


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, 1000 * (time.perf_counter() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="held-out applicants to time")
    args = ap.parse_args()

    # cold start: importing the app and loading every model and explainer
    t0 = time.perf_counter()
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from backend.auth import get_password_hash
    from backend.database import Base, get_db
    from backend.main import app
    from backend.models import Application, User, UserRole
    from backend.services.counterfactual_service import get_counterfactual_service
    from backend.services.explainability_service import get_explainability_service
    from backend.services.prediction_service import get_prediction_service
    from backend.services.report_service import applicant_report

    ps = get_prediction_service()
    xs = get_explainability_service()
    cs = get_counterfactual_service()
    cold_start_ms = 1000 * (time.perf_counter() - t0)

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    client = TestClient(app)
    db = Session()
    db.add(User(email="officer@example.com", password_hash=get_password_hash("pw12345"),
                role=UserRole.OFFICER, full_name="Officer"))
    db.commit()
    client.post("/api/auth/register", json={"email": "applicant@example.com", "password": "pw12345", "full_name": "A"})

    def login(email):
        r = client.post("/api/auth/login", data={"username": email, "password": "pw12345"})
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    A, O = login("applicant@example.com"), login("officer@example.com")

    raw_test = pd.read_csv("data/processed/X_test.csv")
    pick = np.random.default_rng(1).choice(len(raw_test), args.n + 1, replace=False)

    api = {k: [] for k in ("submit", "decide", "read_decision", "applicant_pdf", "what_if", "compliance_pdf", "total")}
    stages = {k: [] for k in ("score", "shap", "lime", "recourse", "applicant_pdf")}
    recourse_by_band = {"REVIEW": [], "REJECT": []}
    bands = []

    for i, row in enumerate(pick):
        warm_up = i == 0
        raw = json.loads(raw_test.iloc[row].to_json())

        # --- through the API
        r, t_submit = timed(lambda: client.post("/api/applications/", json={"features": raw}, headers=A))
        app_id = r.json()["id"]
        r, t_decide = timed(lambda: client.post(f"/api/predictions/{app_id}", headers=A))
        band = r.json()["decision"]
        _, t_read = timed(lambda: client.get(f"/api/applications/{app_id}", headers=A))
        _, t_pdf = timed(lambda: client.get(f"/api/applications/{app_id}/report.pdf", headers=A))
        t_what_if = None
        if band != "APPROVE":
            _, t_what_if = timed(lambda: client.post(f"/api/predictions/{app_id}/recourse",
                                                     json={"locked_levers": ["income"]}, headers=A))
        _, t_comp = timed(lambda: client.get(f"/api/officer/{app_id}/compliance-report.pdf", headers=O))

        # --- stage by stage (same applicant)
        _, t_score = timed(lambda: ps.predict(raw))
        _, t_shap = timed(lambda: xs.generate_shap_explanation(raw))
        _, t_lime = timed(lambda: xs.generate_lime_explanation(raw))
        t_rec = None
        if band != "APPROVE":
            _, t_rec = timed(lambda: cs.generate_counterfactuals(raw))
        session = Session()
        app_row = session.get(Application, app_id)
        _, t_stage_pdf = timed(lambda: applicant_report(app_row))
        session.close()

        if warm_up:
            continue
        bands.append(band)
        for k, v in (("submit", t_submit), ("decide", t_decide), ("read_decision", t_read), ("applicant_pdf", t_pdf),
                     ("what_if", t_what_if), ("compliance_pdf", t_comp),
                     ("total", t_submit + t_decide + t_read + t_pdf)):
            api[k].append(v)
        for k, v in (("score", t_score), ("shap", t_shap), ("lime", t_lime), ("recourse", t_rec),
                     ("applicant_pdf", t_stage_pdf)):
            stages[k].append(v)
        if t_rec is not None:
            recourse_by_band[band].append(t_rec)
        if (i % 25) == 0:
            print(f"  {i}/{args.n}", flush=True)

    result = {
        "applicants": args.n,
        "decisions": {b: bands.count(b) for b in ("APPROVE", "REVIEW", "REJECT")},
        "cold_start_ms": round(cold_start_ms, 1),
        "robust_recourse": cs.ensemble is not None,
        "api": {k: summarize(v) for k, v in api.items()},
        "stages": {k: summarize(v) for k, v in stages.items()},
        "recourse_by_band": {k: summarize(v) for k, v in recourse_by_band.items()},
        "notes": {
            "api.total": "submit + decide + read decision + applicant PDF, i.e. one applicant's full path",
            "api.decide": "scoring, SHAP, LIME, recourse (Review and Reject only) and loan terms, saved to the database",
            "what_if": "recourse recomputed with income locked; Review and Reject only",
        },
        "environment": {"python": platform.python_version(), "machine": platform.machine(),
                        "system": platform.system(), "cpu_count": os.cpu_count()},
    }
    REPORT.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("decisions", "cold_start_ms", "api", "stages")}, indent=2))


if __name__ == "__main__":
    main()
