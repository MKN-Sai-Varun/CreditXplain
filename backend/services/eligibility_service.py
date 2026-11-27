"""
Eligibility rules checked before the model, as a lender's policy rather than a
risk factor:

  minor      an applicant under MIN_ADULT_AGE cannot take on the loan alone.
             A guardian is the co-applicant: the model scores the application
             with the guardian's age, income and employment, an officer
             decides, and the guardian's ID and income proof must be uploaded
             before an officer can approve.
  no_income  the person whose income is assessed has no income, or is a
             student or unemployed. The model still scores the application,
             but it is never approved automatically: an officer decides and
             can ask for a guarantor.

A rule never makes a decision worse than the model's: it only removes
automatic approval (Approve becomes Review). The limits are illustrative
policy settings in backend/config.py.
"""
from typing import Any, Dict, Optional

from backend.config import settings

DAYS_PER_YEAR = 365.25
GUARDIAN_DOCS = ("guardian_id_proof", "guardian_income_proof")

REASONS = {
    "minor": "The applicant is under {age}, so a guardian applies with them. The application is assessed on "
             "the guardian's income and employment, and a loan officer decides.",
    "no_income": "No regular income is recorded for the person being assessed, so the application cannot be "
                 "approved automatically. A loan officer decides and may ask for a guarantor.",
}


def applicant_age(features: Dict[str, Any]) -> Optional[float]:
    days = features.get("DAYS_BIRTH")
    if isinstance(days, bool) or not isinstance(days, (int, float)) or days >= 0:
        return None
    return round(-days / DAYS_PER_YEAR, 2)   # whole days from the form: 18 years arrives as 17.9986


def is_minor(features: Dict[str, Any]) -> bool:
    age = applicant_age(features)
    return age is not None and age < settings.MIN_ADULT_AGE


_REPLACED = {"DAYS_BIRTH", "AGE_YEARS", "AMT_INCOME_TOTAL", "CREDIT_INCOME_RATIO", "NAME_INCOME_TYPE",
             "DAYS_EMPLOYED", "IS_EMPLOYED_ANOMALY"}


def income_type(features: Dict[str, Any]) -> Optional[str]:
    """The income type, given as a category ("Student") or as one-hot columns (NAME_INCOME_TYPE_Student = 1)."""
    if isinstance(features.get("NAME_INCOME_TYPE"), str):
        return features["NAME_INCOME_TYPE"]
    for key, val in features.items():
        if key.startswith("NAME_INCOME_TYPE_") and val == 1:
            return key[len("NAME_INCOME_TYPE_"):].replace("_", " ")
    return None


def assessed_features(features: Dict[str, Any], guardian: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The record the model scores: the applicant's, with a minor's age, income and employment taken from the guardian."""
    if not (is_minor(features) and guardian):
        return dict(features)
    # drop the applicant's values and anything derived from them, so they are rebuilt from the guardian's
    out = {k: v for k, v in features.items()
           if k not in _REPLACED and not k.startswith("NAME_INCOME_TYPE_")}
    out["DAYS_BIRTH"] = -round(guardian["age"] * DAYS_PER_YEAR)
    out["AMT_INCOME_TOTAL"] = float(guardian["yearly_income"])
    out["NAME_INCOME_TYPE"] = guardian["income_type"]
    if guardian.get("years_employed") is not None:
        out["DAYS_EMPLOYED"] = -round(guardian["years_employed"] * DAYS_PER_YEAR)
        out["IS_EMPLOYED_ANOMALY"] = 0
    return out


def _no_income(features: Dict[str, Any]) -> bool:
    income = features.get("AMT_INCOME_TOTAL")
    if isinstance(income, bool) or not isinstance(income, (int, float)) or income <= 0:
        return True
    return income_type(features) in settings.NO_INCOME_TYPES


def check(features: Dict[str, Any], guardian: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Which rules apply, whose details are assessed, and whether automatic approval is allowed."""
    rules = []
    minor = is_minor(features)
    if minor:
        rules.append({"code": "minor", "reason": REASONS["minor"].format(age=settings.MIN_ADULT_AGE)})
    if _no_income(assessed_features(features, guardian)):
        rules.append({"code": "no_income", "reason": REASONS["no_income"]})
    return {
        "rules": rules,
        "assessed_on": "guardian" if minor else "applicant",
        "automatic_approval": not rules,
        "required_documents": list(GUARDIAN_DOCS) if minor else [],
    }


def route(model_decision: str, eligibility: Dict[str, Any]) -> str:
    """Final band: a rule only removes automatic approval."""
    if not eligibility["automatic_approval"] and model_decision == "APPROVE":
        return "REVIEW"
    return model_decision


def recourse_applies(final_decision: str, eligibility: Dict[str, Any]) -> bool:
    """Reject -> Review always helps; Review -> Approve only when automatic approval is possible."""
    if final_decision == "REJECT":
        return True
    return final_decision == "REVIEW" and eligibility["automatic_approval"]
