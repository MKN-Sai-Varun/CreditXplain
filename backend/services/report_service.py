"""
PDF reports for one application.

applicant_report  - decision, main reasons in plain words, recourse options,
                    loan terms and document requests, for the applicant
compliance_report - the same plus model details, LIME rules, recourse
                    preferences, documents, the full audit trail and the
                    latest fairness status, for officers

PDFs are written uncompressed (they stay small) so their text can be checked
directly in tests.
"""
import io
import json
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from xml.sax.saxutils import escape

from backend.config import settings

# plain-language names for the features that most often drive decisions
FEATURE_NAMES = {
    "CREDIT_GOODS_EXT_RISK": "Loan size compared with the goods price, together with credit-bureau scores",
    "EXT_SOURCE_MEAN": "Credit-bureau scores (average)",
    "EXT_SOURCE_1": "Credit-bureau score 1",
    "EXT_SOURCE_2": "Credit-bureau score 2",
    "EXT_SOURCE_3": "Credit-bureau score 3",
    "EXT_SOURCE_MAX": "Best credit-bureau score",
    "EXT_SOURCE_2_3_PRODUCT": "Credit-bureau scores 2 and 3 combined",
    "CREDIT_ANNUITY_RATIO": "Loan amount compared with the yearly repayment (length of repayment)",
    "ANNUITY_EXT_RISK": "Yearly repayment together with credit-bureau scores",
    "ANNUITY_INCOME_RATIO": "Yearly repayment compared with income",
    "CREDIT_INCOME_RATIO": "Loan amount compared with income",
    "CREDIT_TO_GOODS_RATIO": "Loan amount compared with the goods price",
    "AMT_CREDIT": "Requested loan amount",
    "AMT_ANNUITY": "Yearly loan repayment",
    "AMT_GOODS_PRICE": "Price of the goods being financed",
    "AMT_INCOME_TOTAL": "Yearly income",
    "NAME_EDUCATION_TYPE_Higher_education": "Education level (higher education)",
    "DAYS_ID_PUBLISH": "How long ago the ID document was issued",
    "DAYS_LAST_PHONE_CHANGE": "How long ago the phone number changed",
    "DAYS_REGISTRATION": "How long ago the registration was changed",
    "DAYS_EMPLOYED": "Time in current job",
    "EMPLOYMENT_YEARS": "Years in current job",
    "EMPLOYMENT_AGE_RATIO": "Time in current job compared with age",
    "AGE_MINUS_EMPLOYMENT": "Age compared with time in current job",
    "AGE_YEARS": "Age",
    "BUREAU_AVG_CREDIT_AMT": "Average size of past credits",
    "BUREAU_CURRENT_CREDIT_RATIO": "Past credit sizes compared with this loan",
    "PREV_APP_REFUSAL_RATE": "Share of previous applications that were refused",
    "PREV_CURRENT_CREDIT_RATIO": "Previous loan sizes compared with this loan",
    "FLAG_OWN_CAR_Y": "Car ownership",
    "OWN_CAR_AGE": "Age of car",
    "FLAG_WORK_PHONE": "Work phone provided",
    "REGION_POPULATION_RELATIVE": "Population of home region",
    "FLAG_DOCUMENT_3": "Document 3 provided",
    "DOCUMENT_COUNT": "Number of documents provided",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Credit enquiries in the last quarter",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Credit enquiries in the last year",
    "NAME_FAMILY_STATUS_Married": "Family status (married)",
    "NAME_INCOME_TYPE_Working": "Income type (working)",
    "GOODS_CREDIT_RATIO": "Goods price compared with the loan amount",
    "CREDIT_GOODS_GAP": "Difference between the loan amount and the goods price",
    "CREDIT_INCOME_RATIO_NEW": "Loan amount compared with income",
    "GOODS_INCOME_RATIO": "Goods price compared with income",
    "INCOME_PER_FAMILY_MEMBER": "Income per family member",
    "CREDIT_PER_FAMILY_MEMBER": "Loan amount per family member",
    "CREDIT_BURDEN_EXT_RISK": "Repayment burden together with credit-bureau scores",
    "BUREAU_COUNT_INCOME": "Number of past credits, weighted by loan size compared with income",
    "BUREAU_CREDIT_COUNT": "Number of past credits",
    "BUREAU_OVERDUE_CREDIT_RATIO": "Past overdue amounts compared with this loan",
    "PREV_APP_AVG_CREDIT": "Average size of previous loans",
    "PREV_CURRENT_CREDIT_GAP": "Difference between previous loan sizes and this loan",
    "PREV_APP_INCOME_INTENSITY": "Previous loan sizes compared with income",
    "REGISTRATION_AGE_RATIO": "Time since registration change compared with age",
    "FLAG_EMP_PHONE": "Employer phone provided",
    "FLAG_PHONE": "Home phone provided",
    "IS_EMPLOYED_ANOMALY": "Not in employment (e.g. pensioner)",
    "AMT_REQ_CREDIT_BUREAU_HOUR": "Credit enquiries in the last hour",
    "AMT_REQ_CREDIT_BUREAU_DAY": "Credit enquiries in the last day",
    "AMT_REQ_CREDIT_BUREAU_WEEK": "Credit enquiries in the last week",
    "AMT_REQ_CREDIT_BUREAU_MON": "Credit enquiries in the last month",
    "BUREAU_REQUEST_TOTAL": "Total recent credit enquiries",
    "BUREAU_REQUEST_MEAN": "Average recent credit enquiries",
    "BUREAU_REQUEST_MAX": "Most credit enquiries in one period",
    "BUREAU_REQUEST_NONZERO": "Number of periods with credit enquiries",
}

# one-hot category columns: prefix -> readable name ("Education level (lower secondary)")
CATEGORY_NAMES = {
    "NAME_CONTRACT_TYPE": "Loan type",
    "NAME_EDUCATION_TYPE": "Education level",
    "NAME_FAMILY_STATUS": "Family status",
    "NAME_HOUSING_TYPE": "Housing type",
    "NAME_INCOME_TYPE": "Income type",
    "NAME_TYPE_SUITE": "Accompanied by",
    "OCCUPATION_TYPE": "Occupation",
    "ORGANIZATION_TYPE": "Employer type",
    "WEEKDAY_APPR_PROCESS_START": "Application weekday",
    "HOUSETYPE_MODE": "Building type",
    "WALLSMATERIAL_MODE": "Wall material",
    "FONDKAPREMONT_MODE": "Building fund type",
    "EMERGENCYSTATE_MODE": "Building emergency state",
}

DECISION_TEXT = {
    "APPROVE": "Approved",
    "REVIEW": "Referred to a loan officer for review",
    "REJECT": "Not approved at this stage",
}

OUTCOME_TEXT = {
    "REVIEW": "would move the application to officer review",
    "APPROVE": "would make the application eligible for automatic approval",
}

LEVER_TEXT = {
    "credit": "smaller loan",
    "income": "higher verified income",
    "tenure": "longer repayment",
    "wait": "re-apply later",
}

_styles = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=_styles["Heading1"], fontSize=16, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=_styles["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("body", parent=_styles["BodyText"], fontSize=9.5, leading=12.5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10, textColor=colors.grey)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.5, leading=10.5)


def feature_name(feature):
    if feature in FEATURE_NAMES:
        return FEATURE_NAMES[feature]
    if feature.endswith("_LOG") and feature[:-4] in FEATURE_NAMES:
        return FEATURE_NAMES[feature[:-4]]
    for prefix, name in CATEGORY_NAMES.items():
        if feature.startswith(prefix + "_"):
            return f"{name} ({feature[len(prefix) + 1:].replace('_', ' ').lower()})"
    return feature.replace("_", " ").capitalize()


def _size(n):
    return f"{n} bytes" if n < 1024 else f"{n / 1024:.0f} KB"


def money(x):
    return f"INR {x:,.0f}" if x is not None else "-"


def _p(text, style=BODY):
    return Paragraph(escape(str(text)), style)


def _table(rows, widths, header=True):
    data = [[_p(c, CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f5")))
    t.setStyle(TableStyle(style))
    return t


def _render(story, title):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=title, author="CreditXplain",
                            leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            pageCompression=0)
    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------- sections

def _header(application, title):
    out = [_p(title, H1),
           _p(f"Application #{application.id} · submitted {application.created_at:%d %b %Y} · "
              f"report generated {datetime.utcnow():%d %b %Y %H:%M} UTC", SMALL),
           Spacer(1, 4)]
    return out


def _decision(application):
    d = application.decision
    out = [_p("Decision", H2)]
    out.append(_p(f"Decision: {d.decision.value} - {DECISION_TEXT[d.decision.value]}"))
    out.append(_p(f"Estimated chance of repayment difficulty: {d.probability:.1%}"))
    if application.status.value != {"APPROVE": "approved", "REVIEW": "under_review",
                                     "REJECT": "rejected"}[d.decision.value]:
        out.append(_p(f"Current status after officer review: {application.status.value.replace('_', ' ')}"))
    return out


def _reasons(explanation, n=5):
    out = [_p("Main reasons", H2)]
    rows = (explanation.shap_values or []) if explanation else []
    if not rows:
        return out + [_p("No explanation is available for this application.")]
    out.append(_p("The factors that weighed most in this decision, strongest first:"))
    table = [["Factor", "Effect on risk"]]
    for r in rows[:n]:
        table.append([feature_name(r["feature"]), "raises risk" if r["value"] > 0 else "lowers risk"])
    out.append(_table(table, [120 * mm, 40 * mm]))
    return out


def _option_lines(opt):
    lines = []
    for ch in opt.get("changes", []):
        if ch["feature"] == "WAIT_MONTHS":
            lines.append(f"Re-apply after {int(ch['recommended'])} months, staying in the same job "
                         f"and making no new credit applications meanwhile")
        elif ch["feature"] in ("AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY"):
            lines.append(f"{ch['label']}: {money(ch['original'])} to {money(ch['recommended'])}")
    return lines


def _recourse(explanation, detailed=False):
    out = [_p("What could change the decision", H2)]
    options = (explanation.counterfactuals or []) if explanation else []
    if not options:
        return out + [_p("No change within the supported limits (loan amount, income, repayment length, "
                         "waiting up to 12 months) was found that would improve this decision.")]
    out.append(_p("Each option below would move the application to the next stage according to the model. "
                  "These are model-based estimates, not guarantees."))
    for k, opt in enumerate(options, 1):
        levers = ", ".join(LEVER_TEXT.get(lv, lv) for lv in opt.get("levers", []))
        outcome = OUTCOME_TEXT.get(opt["new_decision"], f"leads to {opt['new_decision']}")
        out.append(_p(f"Option {k} ({levers}): {outcome}", ParagraphStyle(
            "opt", parent=BODY, fontName="Helvetica-Bold", spaceBefore=4)))
        for line in _option_lines(opt):
            out.append(_p(f"- {line}"))
        drivers = opt.get("drivers") or []
        if drivers:
            why = "; ".join(feature_name(d["feature"]) for d in drivers)
            out.append(_p(f"Why it helps: {why}", SMALL))
        if detailed:
            out.append(_p(f"New estimated risk {opt['new_probability']:.1%}; margin below cutoff "
                          f"{opt.get('robust_margin', 0):.2f} log-odds; plausible: {opt.get('plausible')}; "
                          f"cost {opt.get('cost', 0):.3f}", SMALL))
    return out


def _loan_terms(application):
    d = application.decision
    terms = d.loan_terms if d else None
    if not terms:
        return []
    rows = [["Loan amount", money(terms.loan_amount)],
            ["Interest rate (annual)", f"{terms.interest_rate:.2%}"],
            ["Tenure", f"{terms.tenure_months} months"],
            ["Monthly installment (EMI)", money(terms.monthly_emi)],
            ["Total payable", money(terms.total_payable)]]
    return [_p("Loan terms", H2), _table(rows, [70 * mm, 60 * mm], header=False),
            _p("Interest rates in this academic system are illustrative and are not an offer.", SMALL)]


def _document_requests(application):
    reqs = [log for log in sorted(application.audit_logs, key=lambda l: l.timestamp)
            if log.action == "request_documents"]
    if not reqs:
        return []
    out = [_p("Documents requested", H2)]
    for log in reqs:
        out.append(_p(f"- {log.timestamp:%d %b %Y}: {log.justification}"))
    return out


# ---------------------------------------------------------------- reports

def applicant_report(application):
    explanation = application.decision.explanation
    story = _header(application, "CreditXplain - Credit Decision Report")
    story += _decision(application)
    story += _reasons(explanation)
    if application.decision.decision.value != "APPROVE":
        story += _recourse(explanation)
    story += _loan_terms(application)
    story += _document_requests(application)
    story += [Spacer(1, 8), _p(
        "This report explains an automated credit-risk assessment. A loan officer reviews cases referred "
        "for review, and you may ask for any decision to be reviewed by a person.", SMALL)]
    return _render(story, f"Credit decision report - application {application.id}")


def _model_info():
    path = settings.MODELS_DIR / "final_model_card.json"
    try:
        card = json.loads(path.read_text())
    except (OSError, ValueError):
        return [["Model card", "not available"]]
    return [["Model", card.get("model_type", "-")],
            ["Features used", str(len(card.get("features_used", [])))],
            ["Excluded features", ", ".join(card.get("features_excluded", [])) or "-"],
            ["Test ROC-AUC", f"{card.get('test_roc_auc', 0):.4f}"],
            ["Approve below / reject from", f"{card.get('approve_threshold', 0):.4f} / "
                                            f"{card.get('reject_threshold', 0):.4f}"]]


def compliance_report(application, audit_logs, fairness=None):
    d = application.decision
    explanation = d.explanation
    story = _header(application, "CreditXplain - Compliance Report")
    story += [_p("Applicant and application", H2), _table([
        ["Applicant user id", str(application.user_id)],
        ["Requested amount", money(application.features_json.get("AMT_CREDIT"))],
        ["Requested tenure", f"{application.tenure_months} months"],
        ["Status", application.status.value.replace("_", " ")],
    ], [70 * mm, 90 * mm], header=False)]
    story += _decision(application)
    story += [_p("Model", H2), _table(_model_info(), [70 * mm, 90 * mm], header=False),
              _p(f"Cutoffs stored with this decision: approve below {d.approve_threshold:.4f}, "
                 f"reject from {d.reject_threshold:.4f}.", SMALL)]

    shap_rows = (explanation.shap_values or []) if explanation else []
    if shap_rows:
        story += [_p("SHAP contributions (log-odds, top 10)", H2), _table(
            [["Feature", "Value", "Contribution"]] +
            [[r["feature"], f"{r['feature_value']:.4g}" if isinstance(r["feature_value"], (int, float))
              else str(r["feature_value"]), f"{r['value']:+.3f}"] for r in shap_rows[:10]],
            [90 * mm, 35 * mm, 35 * mm])]
    lime_rows = (explanation.lime_explanation or []) if explanation else []
    if lime_rows:
        story += [_p("LIME local rules (log-odds)", H2), _table(
            [["Rule", "Weight"]] + [[r.get("rule") or r["feature"], f"{r['weight']:+.3f}"] for r in lime_rows[:10]],
            [125 * mm, 35 * mm])]

    if d.decision.value != "APPROVE":
        story += _recourse(explanation, detailed=True)
    prefs = application.recourse_preferences
    story += [_p("Recourse preferences", H2), _p(
        "None set (all levers allowed)." if not prefs else
        f"Locked levers: {', '.join(prefs.get('locked_levers') or []) or 'none'}; "
        f"maximum loan reduction {prefs.get('max_loan_reduction', 0.5):.0%}; "
        f"maximum wait {prefs.get('max_wait_months', 12)} months.")]
    story += _loan_terms(application)

    docs = list(application.documents)
    story += [_p("Documents", H2)]
    story += [_table([["Uploaded", "Type", "File", "Size"]] +
                     [[f"{doc.uploaded_at:%d %b %Y}", doc.doc_type, doc.original_name,
                       _size(doc.size_bytes)] for doc in docs],
                     [30 * mm, 35 * mm, 70 * mm, 25 * mm])] if docs else [_p("No documents uploaded.")]

    story += [_p("Audit trail", H2)]
    story += [_table([["Time (UTC)", "Action", "Officer", "Status", "Justification"]] +
                     [[f"{log.timestamp:%d %b %Y %H:%M}", log.action.replace("_", " "), str(log.officer_id or "-"),
                       f"{(log.previous_status or '-').replace('_', ' ')} to {(log.new_status or '-').replace('_', ' ')}",
                       log.justification or "-"]
                      for log in audit_logs],
                     [26 * mm, 34 * mm, 14 * mm, 38 * mm, 48 * mm])] if audit_logs else [_p("No audit entries.")]

    story += [_p("Fairness monitoring", H2)]
    if fairness is None:
        story += [_p("No fairness calculation has been run yet.")]
    else:
        fmt = lambda v: f"{v:.3f}" if v is not None else "-"
        story += [_p(f"Latest calculation {fairness.calculated_at:%d %b %Y %H:%M} UTC: age-band DPD "
                     f"{fmt(fairness.age_dpd)}, family-status DPD {fmt(fairness.children_dpd)}. "
                     f"Alert: {fairness.alert_message or 'none'}.")]
    return _render(story, f"Compliance report - application {application.id}")
