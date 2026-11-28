"""
Every table and figure of the paper, built only from the files in reports/
(and models/final_model_card.json), so each number in the paper traces to a file.

Run: python3 -m src.paper_figures
Writes reports/paper/:
  tables/*.tex   IEEE Access table environments, ready for \\input{tables/<name>}
  tables/*.md    the same tables in Markdown, for reading on GitHub
  figures/*.pdf  vector figures with embedded fonts (for the paper)
  figures/*.png  300 dpi copies (for slides and quick viewing)
  numbers.json   the numbers quoted in the text, with their source files
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
REPORTS = ROOT / "reports"
OUT = REPORTS / "paper"

METHOD_NAMES = {
    "hybrid": "Hybrid (deployed)",
    "proposed_with_wait": "Proposed, with waiting",
    "proposed": "Proposed, loan levers only",
    "proposed_ensemble": "Ensemble-robust, 5 of 5",
    "proposed_ensemble_4of5": "Ensemble-robust, 4 of 5",
    "proposed_ensemble_with_wait": "Ensemble-robust, with waiting",
    "proposed_margin_0p2": "Proposed, margin 0.2",
    "ablation_no_margin": "Ablation: no margin",
    "ablation_no_plausibility": "Ablation: no plausibility",
    "random": "Random search",
    "previous": "Baseline engine",
    "dice": "DiCE (20 applicants)",
}
METHOD_ORDER = list(METHOD_NAMES)
TIERS = {"reject_to_review": "Reject to Review", "review_to_approve": "Review to Approve"}
AGE_ORDER = ["<25", "25-35", "35-45", "45-55", "55+"]

# categorical slots 1-5 of the reference palette, in fixed order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLUMN_IN, PAGE_IN = 3.5, 7.16          # IEEE column and page widths


# ---------------------------------------------------------------- loading

def load(reports=REPORTS, card=ROOT / "models" / "final_model_card.json"):
    j = lambda name: json.loads((reports / name).read_text())
    return {
        "card": json.loads(Path(card).read_text()),
        "eda": j("eda/eda_summary.json"),
        "baselines": pd.read_csv(reports / "baseline_model_comparison.csv", index_col=0),
        "bands": pd.read_csv(reports / "final_model_bands.csv"),
        "gender": pd.read_csv(reports / "gender_ablation.csv"),
        "bootstrap": j("bootstrap_models.json"),
        "explanations": j("explanation_consistency.json"),
        "recourse": j("recourse_evaluation.json"),
        "recourse_rows": pd.read_csv(reports / "recourse_evaluation_per_applicant.csv"),
        "latency": j("system_latency.json"),
        "time_cost": j("time_cost_sensitivity.json"),
        "understandability": j("understandability.json"),
        "fairness": pd.read_csv(reports / "fairness/age_band_group_stats.csv"),
        "fairness_summary": pd.read_csv(reports / "fairness/fairness_summary.csv"),
        "age": j("age_fairness_experiment.json"),
    }


def usable_share(rows):
    """Share of applicants given an option that is found, plausible and clears the safety margin."""
    usable = rows["found"].astype(bool) & (rows["plausible"] == True) & (rows["robust"] == True)  # noqa: E712
    return float(usable.mean()) if len(rows) else float("nan")


# ---------------------------------------------------------------- formatting

def pct(x, digits=1):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{digits}f}\\%"


def num(x, digits=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{digits}f}"


def tex_escape(s):
    """Escapes plain text for LaTeX; cells that already hold LaTeX (a backslash or $) are left alone."""
    s = str(s)
    if "\\" in s or "$" in s:
        return s
    for a, b in (("&", "\\&"), ("_", "\\_"), ("#", "\\#"), ("%", "\\%"), ("<", "$<$"), (">", "$>$")):
        s = s.replace(a, b)
    return s


class Table:
    def __init__(self, name, caption, header, rows, wide=False, note=None, align=None):
        self.name, self.caption, self.header, self.rows = name, caption, header, rows
        self.wide, self.note = wide, note
        self.align = align or ("l" + "r" * (len(header) - 1))

    def tex(self):
        env = "table*" if self.wide else "table"
        lines = [f"\\begin{{{env}}}[t]", f"\\caption{{{self.caption}}}", f"\\label{{tab:{self.name}}}",
                 "\\centering", "\\setlength{\\tabcolsep}{4pt}", f"\\begin{{tabular}}{{{self.align}}}", "\\hline",
                 " & ".join(f"\\textbf{{{tex_escape(h)}}}" for h in self.header) + " \\\\", "\\hline"]
        for r in self.rows:
            if r == "---":
                lines.append("\\hline")
            else:
                lines.append(" & ".join(tex_escape(c) for c in r) + " \\\\")
        lines.append("\\hline")
        if self.note:
            lines.append(f"\\multicolumn{{{len(self.header)}}}{{p{{{'0.95' if self.wide else '0.92'}\\linewidth}}}}"
                         f"{{\\footnotesize {self.note}}} \\\\")
        lines += ["\\end{tabular}", f"\\end{{{env}}}", ""]
        return "\n".join(lines)

    def md(self):
        def clean(c):
            c = str(c)
            for a, b in (("\\%", "%"), ("$<$", "<"), ("$>$", ">"), ("\\&", "&"), ("\\geq", "≥"), ("\\pm", "±"),
                         ("``", '"'), ("''", '"'), ("--", "–"), ("$", "")):
                c = c.replace(a, b)
            return c
        out = [f"**{self.caption}**", "", "| " + " | ".join(self.header) + " |",
               "|" + "|".join("---" for _ in self.header) + "|"]
        out += ["| " + " | ".join(clean(c) for c in r) + " |" for r in self.rows if r != "---"]
        if self.note:
            out += ["", clean(self.note)]
        return "\n".join(out) + "\n"


# ---------------------------------------------------------------- tables

def build_tables(d):
    t = []
    eda, card = d["eda"], d["card"]
    t.append(Table("dataset", "Dataset and Evaluation Split", ["Property", "Value"], [
        ["Applicants (Home Credit, application table)", f"{eda['rows']:,}"],
        ["Columns as received", f"{eda['columns']}"],
        ["Default rate", pct(eda["default_rate"], 2)],
        ["Columns missing for more than half of applicants", f"{eda['missing']['columns_over_50pct']}"],
        ["Training / validation / test applicants",
         f"{card['training_samples']:,} / {card['validation_samples']:,} / {card['test_samples']:,}"],
        ["Engineered features used by the model", f"{len(card['features_used'])}"],
        ["Age range (years)", f"{eda['age_min']:.1f}--{eda['age_max']:.1f}"],
    ], align="lr", note="Stratified 80/20 train/test split; validation is 20\\% of the training part. "
                        "Cleaning statistics are learned from the training rows only."))

    b = d["baselines"]
    names = {"logistic_regression": "Logistic regression", "random_forest": "Random forest",
             "xgboost": "XGBoost", "lightgbm": "LightGBM"}
    rows = [[names[m], num(b.loc[m, "roc_auc"]), num(b.loc[m, "thresh_precision"]),
             num(b.loc[m, "thresh_recall"]), num(b.loc[m, "thresh_f1"])] for m in names]
    rows += ["---", [f"Final model (LightGBM, {len(card['features_used'])} features)", num(card["test_roc_auc"]), "--", "--", "--"]]
    t.append(Table("models", "Baselines and Final Model on the Test Set",
                   ["Model", "ROC-AUC", "Precision", "Recall", "F1"], rows,
                   note="Baselines: SMOTE on training data, metrics at each model's F1-optimal threshold. "
                        f"Final model PR-AUC {card['test_pr_auc']:.4f}; gender excluded "
                        f"(cost {d['gender']['test_roc_auc'].iloc[1] - d['gender']['test_roc_auc'].iloc[0]:.4f} ROC-AUC). "
                        f"Ten bootstrap-retrained models: {d['bootstrap']['test_roc_auc_mean']:.4f} $\\pm$ "
                        f"{d['bootstrap']['test_roc_auc_std']:.4f}."))

    bands = d["bands"].set_index("band")
    t.append(Table("bands", "Decision Bands on the Test Set", ["Band", "Rule", "Share", "Actual default", "Mean predicted"], [
        [band.capitalize(), rule, pct(bands.loc[band, "percentage"] / 100), pct(bands.loc[band, "actual_default_rate"]),
         pct(bands.loc[band, "avg_predicted_prob"])]
        for band, rule in (("approve", f"$p < {card['approve_threshold']:.4f}$"),
                           ("review", "between"),
                           ("reject", f"$p \\geq {card['reject_threshold']:.4f}$"))],
        align="llrrr", note="Cutoffs chosen on the validation set (src/decision.py)."))

    e = d["explanations"]
    rows = []
    for label, key in (("Whole-model LIME", "lime_default"), ("Local LIME (proposed)", "lime_local")):
        o = e["overall"][key]
        rows.append([label, num(o["spearman"], 2), pct(o["top5_overlap"], 0), pct(o["sign_agreement_top5"], 0),
                     num(o["locality_gap"], 2), num(o.get("seed_stability_top5", float("nan")), 2)])
    t.append(Table("explanations", "Agreement of LIME With SHAP", ["Set-up", "Spearman", "Top-5 overlap",
                                                                   "Sign agreement", "Locality gap", "Seed stability"],
                   rows, wide=True, note=f"{e['applicants']} test applicants ({e['per_band']} per band), "
                                         f"{e['seeds']} seeds. Spearman on SHAP's top 25 features. Locality gap: rank "
                                         "correlation with the applicant's own SHAP values minus that with another "
                                         "applicant's. Seed stability: top-5 overlap across seeds."))

    rr = d["recourse_rows"]
    for tier, label in TIERS.items():
        s = d["recourse"]["methods"][tier]
        rows = []
        for m in METHOD_ORDER:
            if m not in s:
                continue
            r = s[m]
            rows.append([METHOD_NAMES[m], pct(usable_share(rr[(rr.tier == tier) & (rr.method == m)])),
                         pct(r.get("holds_on_all_retrained")), pct(r.get("usable_and_holds_on_all")),
                         num(r.get("mean_cost"), 3), num(r.get("mean_levers"), 2), num(r.get("latency_ms_mean"), 0)])
        t.append(Table(f"recourse_{tier}", f"Recourse Evaluation, {label}",
                       ["Method", "Usable", "Holds on retrained", "Usable and holds", "Mean change", "Levers",
                        "ms"], rows, wide=True,
                       note="Usable: found, plausible and clears the safety margin (share of applicants). Holds on "
                            "retrained: share of found options that reach the promised band on all 5 held-out "
                            "bootstrap models. 300 applicants per method, DiCE 20."))

    lat = d["latency"]
    stage_names = {"score": "Scoring", "shap": "SHAP", "lime": "Local LIME", "recourse": "Recourse (hybrid)",
                   "applicant_pdf": "Applicant PDF"}
    api_names = {"submit": "Submit application", "decide": "Decide (all stages, saved)", "read_decision": "Read decision",
                 "applicant_pdf": "Applicant PDF", "total": "Applicant's full path", "what_if": "``What if'' recourse",
                 "compliance_pdf": "Compliance PDF"}
    rows = [[stage_names[k], num(v["mean_ms"], 1), num(v["median_ms"], 1), num(v["p95_ms"], 1)]
            for k, v in lat["stages"].items() if v] + ["---"]
    rows += [[api_names[k], num(lat["api"][k]["mean_ms"], 1), num(lat["api"][k]["median_ms"], 1),
              num(lat["api"][k]["p95_ms"], 1)] for k in api_names if lat["api"].get(k)]
    t.append(Table("latency", "Latency (ms)", ["Step", "Mean", "Median", "p95"], rows,
                   note=f"{lat['applicants']} test applicants on a laptop ({lat['environment']['machine']}, "
                        f"Python {lat['environment']['python']}); start-up {lat['cold_start_ms'] / 1000:.1f} s. "
                        "p95: 95th percentile."))

    u = d["understandability"]
    r, a = u["readability"]["all_applicant_text"], u["amount_of_information"]
    t.append(Table("understandability", "Understandability of the Applicant Text", ["Measure", "Value"], [
        ["Flesch reading ease", num(r["flesch_reading_ease"], 1)],
        ["Flesch-Kincaid grade", num(r["fk_grade"], 1)],
        ["Words per sentence", num(r["words_per_sentence"], 1)],
        ["Reasons shown", f"{a['reasons_shown']}"],
        ["Options per Review/Reject applicant (mean)", num(a["options_per_review_or_reject_applicant"]["mean"], 2)],
        ["Change lines per option (mean)", num(a["changes_per_option"]["mean"], 2)],
        ["Words per applicant (median)", num(a["words_per_applicant"]["median"], 0)],
        ["Factors shown with a plain-language name", pct(u["plain_names"]["applicant_view"]["with_written_plain_name"])],
        ["Technical terms in applicant text", f"{sum(u['jargon_in_applicant_text'].values())}"],
    ], align="lr", note=f"{u['applicants_per_band']} test applicants per decision band."))

    v = d["age"]["variants"]
    labels = {"A_current": "Current (age kept)", "B_no_age": "Without direct age", "C_age_free": "Age-free",
              "D_band_cutoffs": "Per-band cutoffs (uses age)"}
    t.append(Table("age_experiment", "Removing Age From the Model", ["Variant", "ROC-AUC", "Approve", "Default if approved",
                                                                     "Age DPD", "Equal opp. gap"],
                   [[labels[k], num(x["test_roc_auc"], 4), pct(x["approve_rate"]), pct(x["default_rate_among_approved"], 2),
                     num(x["demographic_parity_difference"], 3), num(x["equal_opportunity_difference"], 3)]
                    for k, x in v.items()], wide=True,
                   note="Run before cleaning statistics were restricted to training rows (Phase 5.6b); the current "
                        "model's audit gives DPD " + num(float(d["fairness_summary"].set_index("demographic")
                                                              .loc["age_band", "demographic_parity_diff"]), 3) + "."))
    return t


# ---------------------------------------------------------------- figures

def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
    })
    return plt


def _clean(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def _save(fig, name):
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    # no tight cropping: the PDF keeps its exact width (3.5 or 7.16 in), which the IEEE Access class uses as the figure size
    fig.savefig(OUT / "figures" / f"{name}.pdf", metadata={"CreationDate": None})  # no timestamp: reproducible
    fig.savefig(OUT / "figures" / f"{name}.png", dpi=300)


def fig_architecture(plt):
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(PAGE_IN, 2.4), layout="constrained")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def box(x, y, w, h, text, fill="#ffffff", edge=MUTED, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1", fc=fill, ec=edge, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.5,
                fontweight="bold" if bold else "normal", linespacing=1.25)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8,
                                                            shrinkA=0, shrinkB=0))

    box(0.5, 17, 11, 8, "Applicant\nportal")
    box(14.5, 17, 14, 8, "Eligibility rules\n(guardian,\nincome)")
    box(31.5, 17, 13, 8, "LightGBM\nrisk score", fill="#e8f0fb", edge=BLUE, bold=True)
    box(47.5, 17, 12, 8, "Three-band\nrouter")
    box(63.5, 30.5, 14, 7, "Approve:\nloan terms")
    box(63.5, 17.5, 14, 7, "Review")
    box(63.5, 4.5, 14, 7, "Reject")
    box(81.5, 28, 18, 11.5, "Officer review:\ndecision, written\njustification, audit trail")
    box(81.5, 2, 18, 15, "Recourse:\noptions that survive\nretrained models first,\nwaiting lever,\napplicant limits",
        fill="#fdeee6", edge=ORANGE, bold=True)
    box(31.5, 1, 28, 10, "Explanations for every decision:\nSHAP, local LIME,\nplain-language reasons, PDF report")

    arrow(11.8, 21, 14.2, 21)
    arrow(28.8, 21, 31.2, 21)
    arrow(44.8, 21, 47.2, 21)
    arrow(59.8, 22.5, 63.2, 33.5)
    arrow(59.8, 21, 63.2, 21)
    arrow(59.8, 19.5, 63.2, 8.5)
    arrow(77.8, 22, 81.2, 30.5)
    arrow(77.8, 19.5, 81.2, 13)
    arrow(77.8, 8, 81.2, 8)
    arrow(38, 16.7, 38, 11.3)
    _save(fig, "fig1_architecture")
    plt.close(fig)


def fig_recourse(plt, d):
    methods = ["hybrid", "proposed_with_wait", "proposed", "proposed_ensemble", "random", "previous"]
    rr = d["recourse_rows"]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_IN, 2.4), sharey=True, layout="constrained")
    for ax, (tier, label) in zip(axes, TIERS.items()):
        s = d["recourse"]["methods"][tier]
        y = np.arange(len(methods))[::-1]
        usable = [usable_share(rr[(rr.tier == tier) & (rr.method == m)]) * 100 for m in methods]
        holds = [s[m]["usable_and_holds_on_all"] * 100 for m in methods]
        ax.barh(y + 0.19, usable, height=0.36, color=BLUE, label="Usable option")
        ax.barh(y - 0.19, holds, height=0.36, color=ORANGE, label="Usable and survives retraining")
        for yy, u, h in zip(y, usable, holds):
            ax.text(u + 0.8, yy + 0.19, f"{u:.0f}", va="center", fontsize=6.5, color=MUTED)
            ax.text(h + 0.8, yy - 0.19, f"{h:.0f}", va="center", fontsize=6.5, color=MUTED)
        ax.set_yticks(y, [METHOD_NAMES[m] for m in methods])
        ax.set_title(label, loc="left")
        ax.set_xlabel("Applicants (%)")
        ax.set_xlim(0, 80)
        _clean(ax, "x")
    axes[0].legend(loc="lower right", frameon=False)
    _save(fig, "fig2_recourse")
    plt.close(fig)


def fig_robustness(plt, d):
    methods = ["previous", "proposed", "random", "proposed_ensemble", "hybrid"]
    rr = d["recourse_rows"]
    x = rr[(rr.tier == "reject_to_review") & rr.found.astype(bool) & rr.holds_own.notna()]
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.9), layout="constrained")
    shades = ["#d9d8d3", "#b9cfee", "#86b6ef", "#5598e7", "#2a78d6", "#184f95"]   # sequential blue, 0..5 models
    y = np.arange(len(methods))[::-1]
    for yy, m in zip(y, methods):
        h = x[x.method == m].holds_own.astype(int)
        shares = np.bincount(h, minlength=6)[:6] / max(len(h), 1) * 100
        left = 0
        for k, sh in enumerate(shares):
            ax.barh(yy, sh, left=left, color=shades[k], height=0.6, edgecolor="white", linewidth=1,
                    label=f"{k}" if yy == y[0] else None)
            left += sh
    ax.set_yticks(y, [METHOD_NAMES[m] for m in methods])
    ax.set_xlabel("Options found (%)")
    ax.set_xlim(0, 100)
    ax.legend(title="Models (of 5)", ncol=6, loc="upper center", bbox_to_anchor=(0.45, -0.24), frameon=False,
              handlelength=1, columnspacing=0.8, title_fontsize=7)
    _clean(ax, None)
    _save(fig, "fig3_robustness")
    plt.close(fig)


def fig_explanations(plt, d):
    e = d["explanations"]["by_band"]
    bands = ["approve", "review", "reject"]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_IN, 2.0), layout="constrained")
    for ax, (metric, label) in zip(axes, (("spearman", "Rank correlation with SHAP"),
                                          ("sign_agreement_top5", "Sign agreement on SHAP's top 5"))):
        xs = np.arange(len(bands))
        for off, key, color, name in ((-0.18, "lime_default", ORANGE, "Whole-model LIME"),
                                      (0.18, "lime_local", BLUE, "Local LIME")):
            vals = [e[key][b][metric] for b in bands]
            ax.bar(xs + off, vals, width=0.34, color=color, label=name)
        ax.axhline(0, color=MUTED, linewidth=0.6)
        ax.set_xticks(xs, [b.capitalize() for b in bands])
        ax.set_title(label, loc="left")
        ax.set_ylim(-0.2, 1)
        _clean(ax)
    axes[0].legend(frameon=False, loc="upper left")
    _save(fig, "fig4_explanations")
    plt.close(fig)


def fig_age(plt, d):
    f = d["fairness"].set_index("group").loc[AGE_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_IN, 2.0), layout="constrained")
    xs = np.arange(len(AGE_ORDER))
    axes[0].bar(xs, f["actual_default_rate"] * 100, color=ORANGE, width=0.6)
    axes[0].set_title("Actual default rate (%)", loc="left")
    axes[1].bar(xs, f["approve_rate"] * 100, color=BLUE, width=0.6)
    axes[1].set_title("Automatically approved (%)", loc="left")
    for ax, col in zip(axes, ("actual_default_rate", "approve_rate")):
        for x_, v in zip(xs, f[col] * 100):
            ax.text(x_, v + 0.6, f"{v:.1f}", ha="center", fontsize=6.5, color=MUTED)
        ax.set_xticks(xs, AGE_ORDER)
        ax.set_ylim(0, (f[col] * 100).max() * 1.2)
        ax.set_xlabel("Age band (years)")
        _clean(ax)
    _save(fig, "fig5_age")
    plt.close(fig)


def fig_latency(plt, d):
    st = d["latency"]["stages"]
    parts = [("Scoring", st["score"]["mean_ms"], MUTED), ("SHAP", st["shap"]["mean_ms"], AQUA),
             ("Local LIME", st["lime"]["mean_ms"], BLUE), ("Recourse, Review/Reject", st["recourse"]["mean_ms"], ORANGE),
             ("PDF", st["applicant_pdf"]["mean_ms"], YELLOW)]
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 1.9), layout="constrained")
    left = 0
    for name, v, c in parts:
        ax.barh(0, v, left=left, color=c, height=0.5, edgecolor="white", linewidth=1, label=f"{name} ({v:.0f} ms)")
        left += v
    ax.set_yticks([])
    ax.set_xlabel("Mean time per applicant (ms)")
    fig.legend(ncol=2, loc="outside lower center", frameon=False, handlelength=1)   # inside the figure's own area
    _clean(ax, "x")
    ax.spines["left"].set_visible(False)
    _save(fig, "fig6_latency")
    plt.close(fig)


# ---------------------------------------------------------------- numbers quoted in the text

def numbers(d):
    rr, rec = d["recourse_rows"], d["recourse"]["methods"]
    out = {
        "test_roc_auc": round(d["card"]["test_roc_auc"], 4),
        "test_pr_auc": round(d["card"]["test_pr_auc"], 4),
        "approve_below": round(d["card"]["approve_threshold"], 4),
        "reject_from": round(d["card"]["reject_threshold"], 4),
        "bootstrap_auc_mean": round(d["bootstrap"]["test_roc_auc_mean"], 4),
        "lime_local_spearman": d["explanations"]["overall"]["lime_local"]["spearman"],
        "lime_default_spearman": d["explanations"]["overall"]["lime_default"]["spearman"],
        "age_dpd": round(float(d["fairness_summary"].set_index("demographic").loc["age_band", "demographic_parity_diff"]), 4),
        "full_path_ms_mean": d["latency"]["api"]["total"]["mean_ms"],
        "full_path_ms_p95": d["latency"]["api"]["total"]["p95_ms"],
        "flesch_reading_ease": d["understandability"]["readability"]["all_applicant_text"]["flesch_reading_ease"],
    }
    for tier in TIERS:
        for m in ("hybrid", "proposed_with_wait", "proposed", "proposed_ensemble", "random", "previous", "dice"):
            out[f"{tier}.{m}.usable"] = round(usable_share(rr[(rr.tier == tier) & (rr.method == m)]), 4)
            out[f"{tier}.{m}.usable_and_holds"] = rec[tier][m].get("usable_and_holds_on_all")
            out[f"{tier}.{m}.holds_on_all"] = rec[tier][m].get("holds_on_all_retrained")
    return {"numbers": out, "sources": {
        "model": "models/final_model_card.json", "recourse": "reports/recourse_evaluation*.{json,csv}",
        "explanations": "reports/explanation_consistency.json", "fairness": "reports/fairness/",
        "latency": "reports/system_latency.json", "understandability": "reports/understandability.json"}}


def main():
    d = load()
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    tables = build_tables(d)
    for t in tables:
        (OUT / "tables" / f"{t.name}.tex").write_text(t.tex())
        (OUT / "tables" / f"{t.name}.md").write_text(t.md())
    plt = _setup()
    fig_architecture(plt)
    fig_recourse(plt, d)
    fig_robustness(plt, d)
    fig_explanations(plt, d)
    fig_age(plt, d)
    fig_latency(plt, d)
    (OUT / "numbers.json").write_text(json.dumps(numbers(d), indent=2))
    print(f"{len(tables)} tables, 6 figures -> {OUT}")


if __name__ == "__main__":
    main()
