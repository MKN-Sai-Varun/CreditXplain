"""
Turns ORM objects into the dicts the response schemas expect.
"""
from backend.models import Application
from backend.services.report_service import feature_name


def _with_labels(rows):
    return [{**r, "label": feature_name(r["feature"])} for r in (rows or [])]


def labelled_options(options):
    """Recourse options with plain-language labels on their drivers."""
    return [{**o, "drivers": _with_labels(o.get("drivers"))} if o.get("drivers") else o
            for o in (options or [])]


def explanation_detail(explanation) -> dict:
    """Stored explanation with plain-language labels (same wording as the PDF reports)."""
    return {
        "shap_values": _with_labels(explanation.shap_values),
        "lime_explanation": _with_labels(explanation.lime_explanation),
        "counterfactuals": labelled_options(explanation.counterfactuals) if explanation.counterfactuals else None,
    }


def document_requests(application: Application):
    logs = sorted(application.audit_logs, key=lambda log: log.timestamp)
    return [
        {"message": log.justification, "requested_at": log.timestamp}
        for log in logs
        if log.action == "request_documents"
    ]


def application_detail(application: Application) -> dict:
    out = {
        "id": application.id,
        "user_id": application.user_id,
        "status": application.status,
        "created_at": application.created_at,
        "updated_at": application.updated_at,
        "features": application.features_json,
        "tenure_months": application.tenure_months,
        "recourse_preferences": application.recourse_preferences,
        "documents": list(application.documents),
        "document_requests": document_requests(application),
    }
    decision = application.decision
    if decision:
        out["decision"] = decision
        if decision.explanation:
            out["explanation"] = explanation_detail(decision.explanation)
        if decision.loan_terms:
            out["loan_terms"] = decision.loan_terms
    return out
