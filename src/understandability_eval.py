"""
How easy is the applicant-facing text to understand? (Phase 5.4)

Measured without participants, on held-out applicants from every band, using
the same services and wording as the app and the applicant PDF:

  readability   Flesch reading ease and Flesch-Kincaid grade of the text an
                applicant reads (decision, main reasons, options, footer).
                Each line or bullet counts as one sentence, as it is shown.
  amount        reasons, options, changes per option and words an applicant
                has to take in
  plain names   share of displayed factors that have a written plain-language
                name rather than the automatic fallback from the column name,
                and the most frequent fallback names (candidates to rename)
  jargon        technical terms or raw column names left in applicant text

Run: python3 -m src.understandability_eval [--n 50]   (applicants per band)
Writes reports/understandability.json.
"""
import argparse
import json
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPORT = Path("reports/understandability.json")

APPLICANT_REASONS = 5    # DecisionSummary.tsx and the applicant PDF show the top 5
OFFICER_ROWS = 10        # ExplanationTables.tsx shows the top 10 SHAP and LIME rows
JARGON = ("log-odds", "logit", "shap", "lime", "counterfactual", "feature", "threshold", "cutoff",
          "ensemble", "bootstrap", "classifier", "gradient")
RAW_COLUMN = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")   # e.g. EXT_SOURCE_MEAN


# ---------------------------------------------------------------- readability

def syllables(word):
    """Vowel-group count with the usual English adjustments (silent final e, -le, -ed)."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|[^laeiouy]ed|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    return max(1, len(re.findall(r"[aeiouy]+", w)))


def words(text):
    """Words with at least one letter; numbers and amounts such as 'INR 1,20,000' count only the letters."""
    return re.findall(r"[A-Za-z][A-Za-z'\-]*", text)


def readability(sentences):
    """Flesch reading ease and Flesch-Kincaid grade over a list of sentences (None when empty)."""
    sentences = [s for s in sentences if words(s)]
    ws = [w for s in sentences for part in words(s) for w in part.split("-") if w]
    if not sentences or not ws:
        return None
    wps = len(ws) / len(sentences)
    spw = sum(syllables(w) for w in ws) / len(ws)
    return {
        "sentences": len(sentences),
        "words": len(ws),
        "words_per_sentence": round(wps, 2),
        "syllables_per_word": round(spw, 3),
        "flesch_reading_ease": round(206.835 - 1.015 * wps - 84.6 * spw, 1),
        "fk_grade": round(0.39 * wps + 11.8 * spw - 15.59, 1),
    }


# ---------------------------------------------------------------- the applicant's text

def has_plain_name(feature):
    from backend.services.report_service import CATEGORY_NAMES, FEATURE_NAMES
    if feature in FEATURE_NAMES or (feature.endswith("_LOG") and feature[:-4] in FEATURE_NAMES):
        return True
    return any(feature.startswith(p + "_") for p in CATEGORY_NAMES)


def applicant_text(band, probability, shap_rows, options):
    """The lines an applicant reads, in order, with the app's own wording."""
    from backend.services import report_service as rs
    lines = [f"Decision: {band} - {rs.DECISION_TEXT[band]}",
             f"Estimated chance of repayment difficulty: {probability:.1%}",
             rs.REASONS_INTRO]
    for r in shap_rows[:APPLICANT_REASONS]:
        lines.append(f"{rs.feature_name(r['feature'])}: {'raises' if r['value'] > 0 else 'lowers'} risk")
    if band != "APPROVE":
        if not options:
            lines.append(rs.NO_RECOURSE_TEXT)
        else:
            lines.append(rs.RECOURSE_INTRO)
            for k, opt in enumerate(options, 1):
                lines.append(rs.option_heading(k, opt))
                if opt.get("retrain_check_passed") is not None:
                    lines.append(rs.HOLDS_TEXT[bool(opt["retrain_check_passed"])])
                lines += rs.option_lines(opt)
                if opt.get("drivers"):
                    lines.append("Why it helps: " + "; ".join(rs.feature_name(d["feature"]) for d in opt["drivers"]))
    lines.append(rs.FOOTER_TEXT)
    return lines


def jargon_hits(lines):
    text = " ".join(lines)
    low = text.lower()
    hits = Counter({t: len(re.findall(r"\b" + re.escape(t) + r"s?\b", low)) for t in JARGON})
    hits.update({"raw column name": len(RAW_COLUMN.findall(text))})
    return {k: v for k, v in hits.items() if v}


def summarize_counts(values):
    v = np.asarray(values, dtype=float)
    if not len(v):
        return None
    return {"mean": round(float(v.mean()), 2), "median": float(np.median(v)), "max": float(v.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="held-out applicants per decision band")
    args = ap.parse_args()

    import pandas as pd
    from backend.services.counterfactual_service import get_counterfactual_service
    from backend.services.explainability_service import get_explainability_service
    from backend.services.prediction_service import get_prediction_service
    from backend.services.report_service import feature_name, option_lines
    from src.decision import decide

    ps, xs, cs = get_prediction_service(), get_explainability_service(), get_counterfactual_service()
    raw_test = pd.read_csv("data/processed/X_test.csv")
    fe = pd.read_csv("data/processed/X_test_fe_v2.csv")
    sc = ps.scorer
    bands = decide(sc.model.predict(fe[sc.features]), sc.approve_below, sc.reject_from)
    rng = np.random.default_rng(7)
    picks = [(b.upper(), i) for b in ("approve", "review", "reject")
             for i in rng.choice(np.where(bands == b)[0], args.n, replace=False)]

    per_band = {b: [] for b in ("APPROVE", "REVIEW", "REJECT")}
    all_lines = []
    amount = {"options": [], "changes_per_option": [], "levers_per_option": [], "words": []}
    shown = {"applicant": Counter(), "officer": Counter()}
    jargon = Counter()

    for k, (_, i) in enumerate(picks):
        raw = json.loads(raw_test.iloc[i].to_json())
        prob, band = ps.predict(raw)
        shap_rows = xs.generate_shap_explanation(raw)
        lime_rows = xs.generate_lime_explanation(raw)
        options = cs.generate_counterfactuals(raw) if band != "APPROVE" else []
        lines = applicant_text(band, prob, shap_rows, options)

        per_band[band].append(lines)
        all_lines.append(lines)
        jargon.update(jargon_hits(lines))
        amount["words"].append(sum(len(words(s)) for s in lines))
        if band != "APPROVE":
            amount["options"].append(len(options))
            for opt in options:
                amount["changes_per_option"].append(len(option_lines(opt)))   # lines the applicant reads
                amount["levers_per_option"].append(len(opt.get("levers", [])))
        for r in shap_rows[:APPLICANT_REASONS]:
            shown["applicant"][r["feature"]] += 1
        for opt in options:
            for d in opt.get("drivers") or []:
                shown["applicant"][d["feature"]] += 1
        for r in shap_rows[:OFFICER_ROWS] + lime_rows[:OFFICER_ROWS]:
            shown["officer"][r["feature"]] += 1
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(picks)}", flush=True)

    def coverage(counter):
        total = sum(counter.values())
        named = sum(c for f, c in counter.items() if has_plain_name(f))
        fallback = [(f, c) for f, c in counter.most_common() if not has_plain_name(f)]
        return {"factors_shown": total, "distinct_factors": len(counter),
                "with_written_plain_name": round(named / total, 4) if total else None,
                "most_frequent_fallback_names": [{"feature": f, "shown_as": feature_name(f), "times": c}
                                                 for f, c in fallback[:10]]}

    result = {
        "applicants_per_band": args.n,
        "decisions_seen": {b: len(v) for b, v in per_band.items()},
        "readability": {
            "all_applicant_text": readability([s for ls in all_lines for s in ls]),
            "by_decision": {b: readability([s for ls in v for s in ls]) for b, v in per_band.items()},
        },
        "amount_of_information": {
            "reasons_shown": APPLICANT_REASONS,
            "options_per_review_or_reject_applicant": summarize_counts(amount["options"]),
            "changes_per_option": summarize_counts(amount["changes_per_option"]),
            "levers_per_option": summarize_counts(amount["levers_per_option"]),
            "words_per_applicant": summarize_counts(amount["words"]),
        },
        "plain_names": {"applicant_view": coverage(shown["applicant"]),
                        "officer_view": coverage(shown["officer"])},
        "jargon_in_applicant_text": dict(jargon),
        "notes": {
            "readability": "Flesch reading ease: 60-70 plain English, 50-60 fairly difficult, 30-50 difficult. "
                           "FK grade is the US school grade needed. Each displayed line counts as a sentence.",
            "plain_names": "written = an entry in FEATURE_NAMES or CATEGORY_NAMES; otherwise the column name "
                           "is shown with underscores removed",
            "officer_view": "LIME rules shown to officers also contain the raw condition (e.g. 'X <= 0.4'); "
                            "only the factor label is measured here",
        },
    }
    REPORT.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("readability", "amount_of_information", "jargon_in_applicant_text")},
                     indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "most_frequent_fallback_names"}
                      for k, v in result["plain_names"].items()}, indent=2))


if __name__ == "__main__":
    main()
