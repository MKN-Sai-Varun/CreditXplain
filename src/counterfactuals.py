"""
Phase 3A: Counterfactual Recommendation Engine
Generates actionable recourse for rejected applicants using DiCE.

Run: python3 src/counterfactuals.py [--num-samples N] [--num-cfs N]

Options:
  --num-samples: Number of rejected applicants to analyze (default: 10)
  --num-cfs: Number of counterfactuals per applicant (default: 3)
"""
import argparse
import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

import dice_ml
from dice_ml import Dice

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS - ACTIONABLE FEATURES
# =============================================================================

# Based on SHAP importance and domain knowledge
# Actionable: Features an applicant can realistically change
# Immutable: Cannot be changed (age, gender, birth date, etc.)
# Slow-to-change: Can change but requires significant time (education, external scores)

ACTIONABLE_FEATURES = [
    # Financial features (highly actionable)
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    
    # Credit ratios (derived from above)
    "CREDIT_INCOME_RATIO_NEW",
    "ANNUITY_INCOME_RATIO",
    "GOODS_INCOME_RATIO",
    "CREDIT_ANNUITY_RATIO",
    "GOODS_CREDIT_RATIO",
    "CREDIT_GOODS_GAP",
    "CREDIT_TO_GOODS_RATIO",
    "INCOME_PER_FAMILY_MEMBER",
    "CREDIT_PER_FAMILY_MEMBER",
    
    # Employment (somewhat actionable)
    "DAYS_EMPLOYED",
    "EMPLOYMENT_YEARS",
    
    # Family (somewhat actionable - co-applicant)
    "CNT_FAM_MEMBERS",
    
    # Documents (easily actionable)
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    
    # Property ownership (actionable)
    "FLAG_OWN_CAR_Y",
    "OWN_CAR_AGE",
    
    # Contact methods (easily actionable)
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "FLAG_MOBIL",
    "FLAG_CONT_MOBILE",
    "CONTACT_METHOD_COUNT",
    
    # Documents (actionable)
    "DOCUMENT_COUNT",
]

# =============================================================================
# LGBM WRAPPER FOR DiCE
# =============================================================================

class LGBMWrapper:
    """Wrapper to make LightGBM compatible with DiCE"""
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        probs = self.model.predict(X)
        # DiCE expects [1-p, p] format
        return np.column_stack([1 - probs, probs])


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of rejected applicants to analyze")
    parser.add_argument("--num-cfs", type=int, default=3,
                        help="Number of counterfactuals to generate per applicant")
    args = parser.parse_args()

    os.makedirs("reports/counterfactuals", exist_ok=True)

    print("=" * 80)
    print("COUNTERFACTUAL RECOMMENDATION ENGINE")
    print("=" * 80)

    # Load model and data
    print("\nLoading model and data...")
    model = joblib.load("models/final_lgbm.pkl")

    with open("models/final_model_card.json") as f:
        model_card = json.load(f)

    # Load audit test set (has predictions and decisions)
    audit_df = pd.read_csv("data/processed/audit_test.csv")

    # Load original test data to get ground truth demographics
    X_test_original = pd.read_csv("data/processed/X_test_fe_v2.csv")

    print(f"Audit samples: {len(audit_df)}")
    print(f"Features: {len(model_card['features_used'])}")

    # =============================================================================
    # 1. DEFINE ACTIONABLE FEATURES (already defined as constant above)
    # =============================================================================

    print("\n" + "=" * 80)
    print("ACTIONABLE FEATURE CLASSIFICATION")
    print("=" * 80)
    "CREDIT_INCOME_RATIO_NEW",
    "ANNUITY_INCOME_RATIO",
    "GOODS_INCOME_RATIO",
    "CREDIT_ANNUITY_RATIO",
    "GOODS_CREDIT_RATIO",
    "CREDIT_GOODS_GAP",
    "CREDIT_TO_GOODS_RATIO",
    "INCOME_PER_FAMILY_MEMBER",
    "CREDIT_PER_FAMILY_MEMBER",
    
    # Employment (somewhat actionable)
    "DAYS_EMPLOYED",
    "EMPLOYMENT_YEARS",
    
    # Family (somewhat actionable - co-applicant)
    "CNT_FAM_MEMBERS",
    
    # Documents (easily actionable)
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    
    # Property ownership (actionable)
# HELPER FUNCTIONS FOR TESTING
# =============================================================================

def load_counterfactual_model():
    """
    Load the trained model and feature information for counterfactual generation.
    
    Returns:
        tuple: (model, feature_names, continuous_features)
    """
    model = joblib.load("models/final_lgbm.pkl")
    
    with open("models/final_model_card.json") as f:
        model_card = json.load(f)
    
    feature_names = model_card['features_used']
    
    # Load test data to determine continuous features
    X_test = pd.read_csv("data/processed/X_test_fe_v2.csv")
    continuous_features = [f for f in feature_names if X_test[f].nunique() > 10]
    
    return model, feature_names, continuous_features


def generate_counterfactuals(applicant_data, num_counterfactuals=3, diversity_weight=None):
    """
    Generate counterfactuals for a single applicant.
    
    Args:
        applicant_data: DataFrame with single row of applicant features
        num_counterfactuals: Number of counterfactuals to generate
        diversity_weight: Ignored (for API compatibility with tests)
    
    Returns:
        dict: {
            'counterfactuals': DataFrame of counterfactual examples or None,
            'original_prediction': float probability,
            'counterfactual_predictions': list of probabilities,
            'query_instance': original applicant data
        }
    """
    model, feature_names, continuous_features = load_counterfactual_model()
    
    # Ensure applicant data has all required features
    query_instance = applicant_data[feature_names].copy()
    
    # Get original prediction
    original_pred = model.predict(query_instance)[0]
    
    # Load background data for DiCE
    X_background = pd.read_csv("data/processed/X_test_fe_v2.csv")[feature_names]
    
    # Create DiCE data object
    dice_data = dice_ml.Data(
        dataframe=pd.concat([X_background.head(1000), 
                            pd.DataFrame({'outcome': model.predict(X_background.head(1000))})], 
                            axis=1),
        continuous_features=continuous_features,
        outcome_name='outcome'
    )
    
    # Create DiCE model
    wrapped_model = LGBMWrapper(model)
    dice_model = dice_ml.Model(model=wrapped_model, backend='sklearn', model_type='classifier')
    
    # Create DiCE explainer
    dice_explainer = Dice(dice_data, dice_model, method='random')
    
    # Filter actionable features that exist in model
    actionable_in_model = [f for f in ACTIONABLE_FEATURES if f in feature_names]
    
    try:
        # Generate counterfactuals
        cf_examples = dice_explainer.generate_counterfactuals(
            query_instance,
            total_CFs=num_counterfactuals,
            desired_class=0,
            features_to_vary=actionable_in_model,
        )
        
        # Extract counterfactuals
        cf_df = cf_examples.cf_examples_list[0].final_cfs_df
        
        if cf_df is not None and len(cf_df) > 0:
            # Get predictions for counterfactuals
            cf_preds = model.predict(cf_df[feature_names])
            
            return {
                'counterfactuals': cf_df,
                'original_prediction': float(original_pred),
                'counterfactual_predictions': list(cf_preds),
                'query_instance': query_instance
            }
        else:
            return {
                'counterfactuals': None,
                'original_prediction': float(original_pred),
                'counterfactual_predictions': [],
                'query_instance': query_instance
            }
    except Exception as e:
        print(f"Error in generate_counterfactuals: {e}")
        return {
            'counterfactuals': None,
            'original_prediction': float(original_pred),
            'counterfactual_predictions': [],
            'query_instance': query_instance
        }


def rank_counterfactuals_by_feasibility(counterfactuals_df, original_instance):
    """
    Rank counterfactuals by feasibility (fewer changes = more feasible).
    
    Args:
        counterfactuals_df: DataFrame of counterfactual examples
        original_instance: DataFrame with single row of original applicant data
    
    Returns:
        DataFrame: Ranked counterfactuals with feasibility metrics
    """
    ranked = counterfactuals_df.copy()
    original_vals = original_instance.iloc[0]
    
    # Calculate metrics for each counterfactual
    num_changes_list = []
    distance_list = []
    
    for idx, row in ranked.iterrows():
        # Count number of changed features
        changes = 0
        total_distance = 0.0
        
        for col in ranked.columns:
            if col in original_vals.index:
                orig_val = original_vals[col]
                new_val = row[col]
                
                if abs(orig_val - new_val) > 1e-6:
                    changes += 1
                    # Normalized distance
                    col_std = max(ranked[col].std(), 1e-6)
                    total_distance += abs(new_val - orig_val) / col_std
        
        num_changes_list.append(changes)
        distance_list.append(total_distance)
    
    ranked['num_changes'] = num_changes_list
    ranked['distance'] = distance_list
    
    # Feasibility score: inverse of number of changes (normalized)
    ranked['feasibility_score'] = 1.0 / (np.array(num_changes_list) + 1)
    
    # Rank by feasibility (highest first)
    ranked = ranked.sort_values('feasibility_score', ascending=False)
    ranked['rank'] = range(1, len(ranked) + 1)
    
    return ranked


def save_counterfactual_report(ranked_counterfactuals, output_path):
    """
    Save ranked counterfactuals to CSV.
    
    Args:
        ranked_counterfactuals: DataFrame with ranked counterfactuals
        output_path: Path to save CSV file
    """
    ranked_counterfactuals.to_csv(output_path, index=False)
