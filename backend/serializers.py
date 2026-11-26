"""
Turns ORM objects into the dicts the response schemas expect.
"""
from backend.models import Application


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
        "documents": list(application.documents),
        "document_requests": document_requests(application),
    }
    decision = application.decision
    if decision:
        out["decision"] = decision
        if decision.explanation:
            out["explanation"] = decision.explanation
        if decision.loan_terms:
            out["loan_terms"] = decision.loan_terms
    return out
