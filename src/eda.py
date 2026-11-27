"""
Exploratory data analysis of the Home Credit application data, for the paper's
dataset section (KNOWN_ISSUES #1).

Works on the data as received (data/home_credit/application_train.csv), so it
describes the dataset before any cleaning, and notes what src/data_prep.py then
does with it.

Run: python3 -m src.eda
Writes reports/eda/: eda_report.md, eda_summary.json, missing_values.png,
distributions.png, default_rate_by_group.png, top_correlations.csv.
"""
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RAW = Path("data/home_credit/application_train.csv")
OUT = Path("reports/eda")

REPAID, DEFAULTED = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
AGE_BINS = [0, 25, 35, 45, 55, 100]                 # the bands of src/fairness_audit.py
AGE_LABELS = ["<25", "25-35", "35-45", "45-55", "55+"]
EMPLOYED_PLACEHOLDER = 365243                       # DAYS_EMPLOYED code for pensioners / not employed

DISTRIBUTIONS = [
    ("AMT_INCOME_TOTAL", "Yearly income (log10)", lambda s: np.log10(s.clip(lower=1))),
    ("AMT_CREDIT", "Loan amount (log10)", lambda s: np.log10(s.clip(lower=1))),
    ("DAYS_BIRTH", "Age (years)", lambda s: -s / 365.25),
    ("DAYS_EMPLOYED", "Years in current job", lambda s: (-s.where(s != EMPLOYED_PLACEHOLDER)) / 365.25),
    ("EXT_SOURCE_2", "Credit-bureau score 2", lambda s: s),
    ("EXT_SOURCE_3", "Credit-bureau score 3", lambda s: s),
]
GROUPS = [("AGE_BAND", "Age band"), ("NAME_INCOME_TYPE", "Income type"),
          ("NAME_EDUCATION_TYPE", "Education"), ("NAME_FAMILY_STATUS", "Family status")]


# ---------------------------------------------------------------- summaries

def missing_table(df):
    """Columns with missing values: count and share, most missing first."""
    n = df.isna().sum()
    n = n[n > 0].sort_values(ascending=False)
    return pd.DataFrame({"missing": n.astype(int), "share": (n / len(df)).round(4)})


def age_band(days_birth):
    return pd.cut(-pd.Series(days_birth) / 365.25, bins=AGE_BINS, labels=AGE_LABELS).astype(str)


def default_rate_by(df, col, target="TARGET", min_count=0):
    """Applicants and default rate per group, groups smaller than min_count left out."""
    g = df.groupby(col, observed=True)[target].agg(["size", "mean"]).rename(
        columns={"size": "n", "mean": "default_rate"})
    g = g[g["n"] >= min_count]
    g["default_rate"] = g["default_rate"].round(4)
    return g


def top_correlations(df, target="TARGET", k=15):
    """Numeric columns ranked by absolute Spearman correlation with the target."""
    num = df.select_dtypes(include=[np.number]).drop(columns=[target, "SK_ID_CURR"], errors="ignore")
    corr = num.corrwith(df[target], method="spearman").dropna()
    order = corr.abs().sort_values(ascending=False).index[:k]
    return pd.DataFrame({"spearman": corr[order].round(4)})


# ---------------------------------------------------------------- figures

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_missing(missing, path, k=20):
    top = missing.head(k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.barh(top.index, top["share"] * 100, color=REPAID, height=0.7)
    for y, v in enumerate(top["share"] * 100):
        ax.text(v + 0.8, y, f"{v:.0f}%", va="center", fontsize=7, color=MUTED)
    ax.set_xlabel("Applicants with the value missing (%)", fontsize=8, color=INK)
    ax.set_xlim(0, 100)
    ax.set_title(f"Most-missing columns ({len(missing)} of {missing.attrs.get('n_columns', '?')} columns have gaps)",
                 fontsize=9, color=INK, loc="left")
    _style(ax)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_distributions(df, path):
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.2))
    for ax, (col, label, transform) in zip(axes.flat, DISTRIBUTIONS):
        values = transform(df[col])
        lo, hi = np.nanpercentile(values, [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        for target, color, name in ((0, REPAID, "Repaid"), (1, DEFAULTED, "Defaulted")):
            v = values[df["TARGET"] == target].dropna()
            ax.hist(v, bins=bins, density=True, histtype="step", linewidth=2, color=color, label=name)
        ax.set_title(label, fontsize=8.5, color=INK, loc="left")
        ax.set_yticks([])
        _style(ax)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Distributions by outcome (each curve scaled to the same area)", fontsize=9, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_group_rates(tables, overall, path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, (col, label) in zip(axes.flat, GROUPS):
        t = tables[col]
        t = t.iloc[::-1] if col == "AGE_BAND" else t.sort_values("default_rate")   # youngest / riskiest on top
        y = np.arange(len(t))
        ax.barh(y, t["default_rate"] * 100, color=REPAID, height=0.65)
        ax.set_yticks(y, [f"{g}  (n={n:,})" for g, n in zip(t.index, t["n"])], fontsize=7)
        ax.axvline(overall * 100, color=MUTED, linestyle="--", linewidth=1)
        ax.set_title(label, fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("Default rate (%)", fontsize=7.5, color=INK)
        _style(ax)
    fig.suptitle(f"Default rate by group (dashed line: overall {overall:.2%}; groups under 500 applicants left out)",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- report

def write_report(s, path):
    pct = lambda x: f"{x:.2%}"
    lines = [
        "# Exploratory Data Analysis — Home Credit Default Risk",
        "",
        "Generated by `src/eda.py` from `data/home_credit/application_train.csv`, the data as received, before cleaning.",
        "",
        "## Dataset",
        "",
        f"- **{s['rows']:,} applicants**, {s['columns']} columns (including the ID and the target).",
        f"- **Default rate {pct(s['default_rate'])}** ({s['defaults']:,} applicants with payment difficulties): a "
        "strongly imbalanced target.",
        f"- {s['numeric_columns']} numeric and {s['categorical_columns']} categorical columns.",
        f"- Ages {s['age_min']:.1f}–{s['age_max']:.1f} years, so there are no minors (see TECHNICAL_DECISIONS D-76).",
        f"- {s['employed_placeholder']['rows']:,} applicants ({pct(s['employed_placeholder']['share'])}) have the "
        f"placeholder {EMPLOYED_PLACEHOLDER} in `DAYS_EMPLOYED`; "
        f"{pct(s['employed_placeholder']['pensioner_share'])} of them are pensioners. Their default rate is "
        f"{pct(s['employed_placeholder']['default_rate'])}, against {pct(s['employed_placeholder']['others_default_rate'])} "
        "for the rest.",
        "- Split used by every model: stratified 80/20 train/test (`random_state=42`, src/train_baseline.py), then 80/20 "
        "fit/validation inside the training part for early stopping and cutoffs (src/final_model.py).",
        "",
        "## Missing values",
        "",
        f"{s['missing']['columns_with_missing']} of {s['columns']} columns have missing values; "
        f"{s['missing']['columns_over_50pct']} are missing for more than half of the applicants, mostly the building "
        "details (`*_AVG`, `*_MODE`, `*_MEDI`). The credit-bureau scores are missing for "
        f"{pct(s['missing']['EXT_SOURCE_1'])} (score 1), {pct(s['missing']['EXT_SOURCE_2'])} (score 2) and "
        f"{pct(s['missing']['EXT_SOURCE_3'])} (score 3).",
        "",
        "**What src/data_prep.py does:** the `DAYS_EMPLOYED` placeholder becomes a flag (`IS_EMPLOYED_ANOMALY`) and a "
        "missing value; numeric gaps are filled with the column median and categorical gaps with the most common "
        "value; non-binary numeric columns are clipped to the mean ± 3 standard deviations; bureau and "
        "previous-application aggregates are merged in, with 0 for applicants without records.",
        "",
        "![Missing values](missing_values.png)",
        "",
        "## Distributions by outcome",
        "",
        "Defaulters have lower credit-bureau scores, are younger and have shorter job tenure; income and loan amount "
        "differ little between the two groups.",
        "",
        "| Feature | Median, repaid | Median, defaulted |",
        "|---|---|---|",
    ]
    for col, label, _ in DISTRIBUTIONS:
        m = s["medians_by_outcome"][col]
        lines.append(f"| {label.replace(' (log10)', '')} | {m['repaid']:,.2f} | {m['defaulted']:,.2f} |")
    lines += ["", "![Distributions](distributions.png)", "", "## Default rate by group", ""]
    for col, label in GROUPS:
        t = s["default_rate_by_group"][col]
        parts = ", ".join(f"{g} {pct(v['default_rate'])}" for g, v in t.items())
        lines.append(f"- **{label}:** {parts}")
    lines += [
        "",
        f"The age pattern (default falling from {pct(s['default_rate_by_group']['AGE_BAND']['<25']['default_rate'])} "
        f"under 25 to {pct(s['default_rate_by_group']['AGE_BAND']['55+']['default_rate'])} over 55) is what the "
        "model's age-band approval gap follows (`reports/age_fairness_experiment.json`).",
        "",
        "![Default rate by group](default_rate_by_group.png)",
        "",
        "## Numeric columns most related to default",
        "",
        "Ranked by absolute Spearman correlation with the target, each over the applicants who have the value "
        "(full table in `top_correlations.csv`). `DAYS_*` columns count days before the application as negative "
        "numbers, so a positive sign means more recent: younger applicants, newer phone numbers and ID documents "
        "go with more defaults.",
        "",
        "| Column | Spearman |",
        "|---|---|",
    ]
    lines += [f"| `{c}` | {v:+.3f} |" for c, v in s["top_correlations"].items()]
    lines += ["", "The credit-bureau scores are by far the strongest single signals, as in the final model's SHAP "
                  "ranking (`reports/shap/`).", ""]
    path.write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW)
    df["AGE_BAND"] = age_band(df["DAYS_BIRTH"]).values

    raw_cols = [c for c in df.columns if c != "AGE_BAND"]
    missing = missing_table(df[raw_cols])
    missing.attrs["n_columns"] = len(raw_cols)
    placeholder = df["DAYS_EMPLOYED"] == EMPLOYED_PLACEHOLDER
    overall = float(df["TARGET"].mean())

    tables = {col: default_rate_by(df, col, min_count=500) for col, _ in GROUPS}
    tables["AGE_BAND"] = tables["AGE_BAND"].reindex(AGE_LABELS)
    corr = top_correlations(df[raw_cols])
    corr.to_csv(OUT / "top_correlations.csv")
    medians = {}
    for col, _, transform in DISTRIBUTIONS:
        v = transform(df[col]) if col in ("DAYS_BIRTH", "DAYS_EMPLOYED") else df[col]
        medians[col] = {"repaid": float(v[df["TARGET"] == 0].median()),
                        "defaulted": float(v[df["TARGET"] == 1].median())}

    summary = {
        "rows": int(len(df)),
        "columns": len(raw_cols),
        "defaults": int(df["TARGET"].sum()),
        "default_rate": round(overall, 4),
        "numeric_columns": int(df[raw_cols].select_dtypes(include=[np.number]).shape[1]),
        "categorical_columns": int(df[raw_cols].select_dtypes(exclude=[np.number]).shape[1]),
        "age_min": round(float((-df["DAYS_BIRTH"] / 365.25).min()), 1),
        "age_max": round(float((-df["DAYS_BIRTH"] / 365.25).max()), 1),
        "employed_placeholder": {
            "rows": int(placeholder.sum()),
            "share": round(float(placeholder.mean()), 4),
            "pensioner_share": round(float((df.loc[placeholder, "NAME_INCOME_TYPE"] == "Pensioner").mean()), 4),
            "default_rate": round(float(df.loc[placeholder, "TARGET"].mean()), 4),
            "others_default_rate": round(float(df.loc[~placeholder, "TARGET"].mean()), 4),
        },
        "missing": {
            "columns_with_missing": int(len(missing)),
            "columns_over_50pct": int((missing["share"] > 0.5).sum()),
            **{c: float(missing["share"].get(c, 0.0)) for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")},
            "top_20": {c: float(v) for c, v in missing["share"].head(20).items()},
        },
        "medians_by_outcome": medians,
        "default_rate_by_group": {col: {str(g): {"n": int(r.n), "default_rate": float(r.default_rate)}
                                        for g, r in t.iterrows()} for col, t in tables.items()},
        "top_correlations": {c: float(v) for c, v in corr["spearman"].items()},
    }
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    plot_missing(missing, OUT / "missing_values.png")
    plot_distributions(df, OUT / "distributions.png")
    plot_group_rates(tables, overall, OUT / "default_rate_by_group.png")
    write_report(summary, OUT / "eda_report.md")
    print(json.dumps({k: summary[k] for k in ("rows", "columns", "default_rate", "employed_placeholder")}, indent=2))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
