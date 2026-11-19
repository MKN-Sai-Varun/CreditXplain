"""
Measures recourse quality on held-out rejected applicants.
Run: python3 -m src.recourse_validity [N]
"""
import json
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

from src.decision import decide
from src.inference import Scorer
from src.recourse import BAND, apply_changes, find_recourse

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300

sc = Scorer()
base = pd.read_csv("data/processed/X_test.csv")
fe = pd.read_csv("data/processed/X_test_fe_v2.csv")
probs = sc.model.predict(fe[sc.features])
dec = decide(probs, sc.approve_below, sc.reject_from)

rejects = np.where(dec == "reject")[0]
pick = np.random.default_rng(42).choice(rejects, size=min(N, len(rejects)), replace=False)

rows, lat = [], []
lever_counts = Counter()
verified = total = 0

for i in pick:
    raw = base.iloc[i].to_dict()
    t0 = time.perf_counter()
    res = find_recourse(sc, raw)
    lat.append(time.perf_counter() - t0)

    for cf in res:
        total += 1
        out = sc.score(apply_changes(raw, cf["changes"]))
        if BAND[out["decision"]] < BAND["reject"] and abs(out["probability"] - cf["new_probability"]) < 1e-6:
            verified += 1
        lever_counts.update(cf["levers"])

    rows.append({
        "row": int(i),
        "has_recourse": len(res) > 0,
        "reaches_approve": any(cf["reaches_approve"] for cf in res),
        "best_cost": res[0]["cost"] if res else None,
        "best_levers": "+".join(res[0]["levers"]) if res else "",
    })

df = pd.DataFrame(rows)
summary = {
    "rejected_applicants_evaluated": int(len(df)),
    "recourse_found_rate": round(float(df["has_recourse"].mean()), 4),
    "reaches_approve_rate": round(float(df["reaches_approve"].mean()), 4),
    "options_generated": int(total),
    "options_verified_by_rescore": int(verified),
    "validity_rate": round(verified / total, 4) if total else None,
    "mean_best_cost": round(float(df["best_cost"].mean()), 4) if df["best_cost"].notna().any() else None,
    "lever_usage": dict(lever_counts),
    "latency_ms_mean": round(1000 * float(np.mean(lat)), 1),
    "latency_ms_p95": round(1000 * float(np.percentile(lat, 95)), 1),
}

df.to_csv("reports/recourse_validity_per_applicant.csv", index=False)
with open("reports/recourse_validity.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))