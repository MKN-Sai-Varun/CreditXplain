"""
Readability and wording checks used by src/understandability_eval.py. No model or data needed.
"""
import pytest

from src.understandability_eval import (applicant_text, has_plain_name, jargon_hits, readability,
                                        syllables, words)


@pytest.mark.parametrize("word,n", [("the", 1), ("loan", 1), ("make", 1), ("table", 2), ("credit", 2),
                                    ("income", 2), ("approved", 2), ("repayment", 3), ("application", 4)])
def test_syllables(word, n):
    assert syllables(word) == n


def test_words_skip_numbers():
    assert words("INR 1,20,000 to INR 90,000") == ["INR", "to", "INR"]
    assert syllables("1,20,000") == 0


def test_readability_hand_example():
    # 6 one-syllable words in one sentence: 206.835 - 1.015*6 - 84.6*1 = 116.145; grade 0.39*6 + 11.8 - 15.59 = -1.45
    r = readability(["The cat sat on the mat."])
    assert r["words"] == 6 and r["sentences"] == 1
    assert r["flesch_reading_ease"] == pytest.approx(116.1, abs=0.06)
    assert r["fk_grade"] == pytest.approx(-1.4, abs=0.06)


def test_readability_counts_hyphen_parts_and_skips_empty_lines():
    r = readability(["Credit-bureau score", "12,000", ""])
    assert r["sentences"] == 1 and r["words"] == 3
    assert readability([]) is None and readability(["42%"]) is None


def test_jargon_and_raw_columns():
    hits = jargon_hits(["EXT_SOURCE_MEAN raises risk", "SHAP features in log-odds"])
    assert hits == {"raw column name": 1, "shap": 1, "feature": 1, "log-odds": 1}
    assert jargon_hits(["Loan amount compared with income: raises risk"]) == {}


def test_plain_name_rule():
    assert has_plain_name("EXT_SOURCE_MEAN")
    assert has_plain_name("AMT_CREDIT_LOG")                 # log of a named column
    assert has_plain_name("OCCUPATION_TYPE_Laborers")       # one-hot category
    assert not has_plain_name("SOME_UNNAMED_COLUMN")


def test_applicant_text_uses_app_wording():
    from backend.services import report_service as rs
    shap_rows = [{"feature": "EXT_SOURCE_MEAN", "value": 0.8}, {"feature": "AMT_INCOME_TOTAL", "value": -0.2}]
    option = {"levers": ["credit"], "new_decision": "REVIEW", "retrain_check_passed": True,
              "changes": [{"feature": "AMT_CREDIT", "label": "Loan amount", "original": 500000.0,
                           "recommended": 400000.0}],
              "drivers": [{"feature": "CREDIT_INCOME_RATIO"}]}
    lines = applicant_text("REJECT", 0.21, shap_rows, [option])
    assert lines[0] == "Decision: REJECT - " + rs.DECISION_TEXT["REJECT"]
    assert "Credit-bureau scores (average): raises risk" in lines
    assert "Yearly income: lowers risk" in lines
    assert rs.RECOURSE_INTRO in lines and rs.HOLDS_TEXT[True] in lines
    assert "Option 1 (smaller loan): " + rs.OUTCOME_TEXT["REVIEW"] in lines
    assert "Why it helps: Loan amount compared with income" in lines
    assert lines[-1] == rs.FOOTER_TEXT

    no_options = applicant_text("REVIEW", 0.08, shap_rows, [])
    assert rs.NO_RECOURSE_TEXT in no_options
    approved = applicant_text("APPROVE", 0.02, shap_rows, [])
    assert rs.NO_RECOURSE_TEXT not in approved and rs.RECOURSE_INTRO not in approved
