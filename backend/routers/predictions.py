from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import User, Application, Decision, Explanation, ApplicationStatus, DecisionType
from backend.auth import get_current_user
from backend.schemas import DecisionResponse
from backend.services.prediction_service import get_prediction_service
from backend.services.explainability_service import get_explainability_service
from backend.services.counterfactual_service import get_counterfactual_service
from backend.services.loan_calculator import loan_terms_for

router = APIRouter()

@router.post("/{application_id}", response_model=DecisionResponse)
def predict_application(
    application_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate prediction, explanation, and counterfactuals for an application.
    
    Pipeline:
    1. Load application features
    2. Make prediction with LightGBM model
    3. Apply three-band decision router
    4. Generate SHAP explanation
    5. Generate LIME explanation
    6. Generate counterfactuals if rejected
    7. Save decision and explanations to database
    8. Update application status
    """
    # Get application
    application = db.query(Application).filter(
        Application.id == application_id
    ).first()
    
    if not application:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found"
        )
    
    # Check authorization
    if current_user.role == "applicant" and application.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to predict this application"
        )
    
    # Check if already predicted
    if application.decision:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Application already has a prediction"
        )

    amount = application.features_json.get("AMT_CREDIT")
    if not isinstance(amount, (int, float)) or amount <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Application has no valid requested loan amount (AMT_CREDIT)"
        )
    
    try:
        # Get services
        prediction_service = get_prediction_service()
        explainability_service = get_explainability_service()
        counterfactual_service = get_counterfactual_service()
        
        # Step 1: Make prediction
        probability, decision_str = prediction_service.predict(application.features_json)
        
        # Step 2: Create decision record
        decision = Decision(
            application_id=application.id,
            decision=DecisionType[decision_str],
            probability=probability,
            approve_threshold=prediction_service.approve_below,
            reject_threshold=prediction_service.reject_from
        )
        db.add(decision)
        db.flush()  # Get decision ID
        
        # Step 3: Generate SHAP explanation
        shap_values = explainability_service.generate_shap_explanation(application.features_json)
        
        # Step 4: Generate LIME explanation
        lime_explanation = explainability_service.generate_lime_explanation(application.features_json)
        
        # Step 5: Recourse for anyone not approved (Reject -> Review, Review -> Approve)
        counterfactuals = None
        if decision_str in ("REJECT", "REVIEW"):
            counterfactuals = counterfactual_service.generate_counterfactuals(application.features_json)
        
        # Step 6: Create explanation record
        explanation = Explanation(
            decision_id=decision.id,
            shap_values=shap_values,
            lime_explanation=lime_explanation,
            counterfactuals=counterfactuals
        )
        db.add(explanation)
        
        # Step 7: Loan terms if approved, over the tenure the applicant asked for
        if decision_str == "APPROVE":
            db.add(loan_terms_for(application, decision))
        
        # Step 8: Update application status
        if decision_str == "APPROVE":
            application.status = ApplicationStatus.APPROVED
        elif decision_str == "REVIEW":
            application.status = ApplicationStatus.UNDER_REVIEW
        else:  # REJECT
            application.status = ApplicationStatus.REJECTED
        
        # Commit all changes
        db.commit()
        db.refresh(decision)
        
        return decision
    
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
