"""
The paper's tables must match the report files they come from.
Uses only committed files (reports/, models/final_model_card.json), so it also runs in CI.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.paper_figures import Table, build_tables, load, pct, tex_escape, usable_share

NEEDED = [Path("models/final_model_card.json"), Path("reports/recourse_evaluation_per_applicant.csv"),
          Path("reports/eda/eda_summary.json"), Path("reports/age_fairness_experiment.json")]


def test_tex_escape_and_percent():
    assert tex_escape("A_B & C #1") == "A\\_B \\& C \\#1"
    assert tex_escape("<25") == "$<$25"
    assert tex_escape("$p < 0.03$") == "$p < 0.03$"          # already LaTeX: untouched
    assert pct(0.2367) == "23.7\\%" and pct(None) == "--"


def test_table_renders_tex_and_markdown():
    t = Table("demo", "Demo Table", ["Method", "Usable"], [["Hybrid (deployed)", pct(0.62)], "---", ["x_y", "1"]],
              note="A note.")
    tex = t.tex()
    assert "\\begin{table}[t]" in tex and "\\label{tab:demo}" in tex
    assert "Hybrid (deployed) & 62.0\\% \\\\" in tex and "x\\_y & 1 \\\\" in tex
    assert tex.count("\\hline") == 4
    md = t.md()
    assert "| Hybrid (deployed) | 62.0% |" in md and "A note." in md
    assert Table("w", "W", ["a", "b"], [], wide=True).tex().startswith("\\begin{table*}")


def test_usable_share():
    rows = pd.DataFrame({"found": [True, True, False, True], "plausible": [True, False, None, True],
                         "robust": [True, True, None, False]})
    assert usable_share(rows) == 0.25


@pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="report files missing")
def test_tables_match_their_sources():
    d = load()
    tables = {t.name: t for t in build_tables(d)}
    assert len(tables) == 9

    rec = json.loads(Path("reports/recourse_evaluation.json").read_text())["methods"]
    rr = pd.read_csv("reports/recourse_evaluation_per_applicant.csv")
    for tier in ("reject_to_review", "review_to_approve"):
        rows = {r[0]: r for r in tables[f"recourse_{tier}"].rows if r != "---"}
        hybrid, src = rows["Hybrid (deployed)"], rec[tier]["hybrid"]
        assert hybrid[1] == pct(usable_share(rr[(rr.tier == tier) & (rr.method == "hybrid")]))
        assert hybrid[2] == pct(src["holds_on_all_retrained"])
        assert hybrid[3] == pct(src["usable_and_holds_on_all"])
        assert len(rows) == len(rec[tier])                     # every evaluated method is in the table

    card = json.loads(Path("models/final_model_card.json").read_text())
    final = next(r for r in tables["models"].rows if r != "---" and r[0].startswith("Final model"))
    assert final[1] == f"{card['test_roc_auc']:,.3f}"
    assert f"{card['approve_threshold']:.4f}" in tables["bands"].rows[0][1]

    lat = json.loads(Path("reports/system_latency.json").read_text())
    full = next(r for r in tables["latency"].rows if r != "---" and r[0] == "Applicant's full path")
    assert full[1] == f"{lat['api']['total']['mean_ms']:,.1f}"

    for t in tables.values():                                 # every table compiles to one environment
        tex = t.tex()
        assert tex.count("\\begin{table") == 1 and tex.count("\\end{table") == 1
        assert all(len(r) == len(t.header) for r in t.rows if r != "---")
