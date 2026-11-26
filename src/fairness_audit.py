"""
Phase 3B: Fairness and Bias Audit
Analyzes demographic disparities in model predictions and decisions.

Run: python3 src/fairness_audit.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    MetricFrame
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings('ignore')

# =============================================================================
# HELPER FUNCTIONS FOR DEMOGRAPHIC ANALYSIS
# =============================================================================

def create_age_bands(df):
    """Create age bands from DAYS_BIRTH or AGE_YEARS."""
    df = df.copy()
    
    # Check if age info already exists
    if 'AGE_YEARS' in df.columns:
        pass  # Already have it
    elif 'DAYS_BIRTH' in df.columns:
        df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365.25
    else:
        # Load from original test data
        X_test_full = pd.read_csv("data/processed/X_test_fe_v2.csv")
        if 'DAYS_BIRTH' in X_test_full.columns:
            # Align indices
            df['AGE_YEARS'] = (-X_test_full.loc[df.index, 'DAYS_BIRTH'] / 365.25).values
        else:
            # No age data available, return without age bands
            df['age_band'] = 'Unknown'
            return df
    
    df['age_band'] = pd.cut(df['AGE_YEARS'], 
                             bins=[0, 25, 35, 45, 55, 100],
                             labels=['<25', '25-35', '35-45', '45-55', '55+'])
    return df

def create_family_status(df):
    """Create family status categories."""
    df = df.copy()
    # Simple categorization based on marital status and children
    if 'NAME_FAMILY_STATUS_Married' in df.columns:
        df['family_status'] = 'Single'
        df.loc[df['NAME_FAMILY_STATUS_Married'] == 1, 'family_status'] = 'Married'
        if 'CNT_CHILDREN' in df.columns:
            df.loc[df['CNT_CHILDREN'] > 0, 'family_status'] = 'Has Children'
    else:
        df['family_status'] = 'Unknown'
    return df

def compute_demographic_parity(df, sensitive_feature):
    """Compute demographic parity difference."""
    try:
        dp_diff = demographic_parity_difference(
            y_true=df['y_true'],
            y_pred=df['y_pred'],
            sensitive_features=df[sensitive_feature]
        )
        return float(dp_diff)
    except Exception as e:
        return None

def compute_equalized_odds(df, sensitive_feature):
    """Compute equalized odds difference."""
    try:
        eo_diff = equalized_odds_difference(
            y_true=df['y_true'],
            y_pred=df['y_pred'],
            sensitive_features=df[sensitive_feature]
        )
        return float(eo_diff)
    except Exception as e:
        return None

def compute_fairness_metrics(df, sensitive_feature):
    """Compute comprehensive fairness metrics for a sensitive feature."""
    
    # Overall metrics
    dp_diff = compute_demographic_parity(df, sensitive_feature)
    eo_diff = compute_equalized_odds(df, sensitive_feature)
    
    # Group-level metrics
    groups = df.groupby(sensitive_feature, observed=False)
    
    group_metrics = []
    for group_name, group_df in groups:
        n = len(group_df)
        positive_rate = group_df['y_pred'].mean()
        default_rate = group_df['y_true'].mean()
        
        group_metrics.append({
            'group': group_name,
            'count': n,
            'positive_rate': positive_rate,
            'default_rate': default_rate
        })
    
    group_metrics_df = pd.DataFrame(group_metrics)
    
    return {
        'demographic_parity_difference': dp_diff,
        'equalized_odds_difference': eo_diff,
        'group_metrics': group_metrics_df
    }

def load_fairness_data():
    """Load audit test data with predictions for fairness analysis."""
    audit_df = pd.read_csv("data/processed/audit_test.csv")
    
    # Ensure required columns exist
    if 'predicted_probability' in audit_df.columns and 'y_pred' not in audit_df.columns:
        audit_df['y_pred'] = (audit_df['predicted_probability'] >= 0.5).astype(int)
    if 'actual_default' in audit_df.columns and 'y_true' not in audit_df.columns:
        audit_df['y_true'] = audit_df['actual_default']
    
    # Add default_probability alias if needed
    if 'predicted_probability' in audit_df.columns and 'default_probability' not in audit_df.columns:
        audit_df['default_probability'] = audit_df['predicted_probability']
    
    return audit_df

def generate_fairness_report(df, protected_attributes, output_path=None):
    """Generate comprehensive fairness report for multiple protected attributes."""
    report = {}
    
    for attr in protected_attributes:
        if attr in df.columns and df[attr].notna().sum() > 0:
            metrics = compute_fairness_metrics(df, attr)
            report[attr] = metrics
    
    if output_path:
        import json
        with open(output_path, 'w') as f:
            # Convert DataFrames to dict for JSON serialization
            json_report = {}
            for attr, metrics in report.items():
                json_report[attr] = {
                    'demographic_parity_difference': metrics['demographic_parity_difference'],
                    'equalized_odds_difference': metrics['equalized_odds_difference'],
                    'group_metrics': metrics['group_metrics'].to_dict(orient='records')
                }
            json.dump(json_report, f, indent=2)
    
    return report


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    os.makedirs("reports/fairness", exist_ok=True)

    print("=" * 80)
    print("FAIRNESS AND BIAS AUDIT")
    print("=" * 80)

    # =============================================================================
    # 1. LOAD DATA
    # =============================================================================

    print("\nLoading data...")

    # Load audit test set (has predictions and decisions)
    audit_df = pd.read_csv("data/processed/audit_test.csv")

    # Load original test features to get demographics
    X_test_full = pd.read_csv("data/processed/X_test_fe_v2.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    print(f"Audit samples: {len(audit_df)}")

    # =============================================================================
    # 2. RECONSTRUCT DEMOGRAPHIC FEATURES
    # =============================================================================

    print("\n" + "=" * 80)
    print("DEMOGRAPHIC FEATURE RECONSTRUCTION")
    print("=" * 80)

    # Extract demographics from original features (not in model)
    demographics = pd.DataFrame(index=audit_df.index)

    # Age bands
    if 'AGE_YEARS' in X_test_full.columns:
        demographics['age_years'] = X_test_full['AGE_YEARS']
        demographics['age_band'] = pd.cut(
            demographics['age_years'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['<25', '25-35', '35-45', '45-55', '55+'],
            include_lowest=True
        )
    else:
        # Derive from DAYS_BIRTH if AGE_YEARS not present
        if 'DAYS_BIRTH' in X_test_full.columns:
            demographics['age_years'] = (-X_test_full['DAYS_BIRTH'] / 365).astype(int)
            demographics['age_band'] = pd.cut(
                demographics['age_years'],
                bins=[0, 25, 35, 45, 55, 100],
                labels=['<25', '25-35', '35-45', '45-55', '55+'],
                include_lowest=True
            )

    # Gender (for monitoring only - not in model)
    gender_cols = [c for c in X_test_full.columns if 'CODE_GENDER' in c.upper()]
    if gender_cols:
        # Reconstruct gender from one-hot encoding
        if 'CODE_GENDER_M' in X_test_full.columns:
            demographics['gender'] = X_test_full['CODE_GENDER_M'].map({1: 'M', 0: 'F/Other'})
        elif 'CODE_GENDER_F' in X_test_full.columns:
            demographics['gender'] = X_test_full['CODE_GENDER_F'].map({1: 'F', 0: 'M/Other'})
        else:
            demographics['gender'] = 'Unknown'
    else:
        demographics['gender'] = 'Unknown'

    # Family status
    if 'NAME_FAMILY_STATUS_Married' in X_test_full.columns:
        demographics['family_status'] = X_test_full['NAME_FAMILY_STATUS_Married'].map({
            1: 'Married', 0: 'Not Married'
        })
    elif 'CNT_FAM_MEMBERS' in X_test_full.columns:
        demographics['family_status'] = (X_test_full['CNT_FAM_MEMBERS'] > 1).map({
            True: 'Family (2+)', False: 'Single'
        })
    else:
        demographics['family_status'] = 'Unknown'

    # Children
    if 'CNT_CHILDREN' in X_test_full.columns:
        demographics['has_children'] = (X_test_full['CNT_CHILDREN'] > 0).map({
            True: 'With Children', False: 'No Children'
        })
    else:
        demographics['has_children'] = 'Unknown'

    # Education
    education_cols = [c for c in X_test_full.columns if 'NAME_EDUCATION_TYPE' in c]
    if education_cols:
        if 'NAME_EDUCATION_TYPE_Higher_education' in X_test_full.columns:
            demographics['education'] = X_test_full['NAME_EDUCATION_TYPE_Higher_education'].map({
                1: 'Higher Education', 0: 'Other Education'
            })
        else:
            demographics['education'] = 'Unknown'
    else:
        demographics['education'] = 'Unknown'

    print("\nDemographic features reconstructed:")
    for col in demographics.columns:
        if col not in ['age_years']:
            print(f"  {col}: {demographics[col].nunique()} categories")

    # =============================================================================
    # 3. PREPARE METRICS DATA
    # =============================================================================

    print("\n" + "=" * 80)
    print("PREPARING FAIRNESS METRICS")
    print("=" * 80)

    # Get predictions and decisions
    y_true = y_test.values
    y_pred_prob = audit_df['predicted_probability'].values
    y_pred_binary = (audit_df['decision'] != 'approve').astype(int)  # 0=approve, 1=review/reject

    print(f"\nActual default rate: {y_true.mean():.2%}")
    print(f"Predicted high risk rate: {y_pred_binary.mean():.2%}")
    print(f"Approve rate: {(audit_df['decision'] == 'approve').mean():.2%}")

    # =============================================================================
    # 4. FAIRNESS METRICS BY DEMOGRAPHIC GROUP
    # =============================================================================

    print("\n" + "=" * 80)
    print("FAIRNESS METRICS ANALYSIS")
    print("=" * 80)

    fairness_results = []

    # Analyze each demographic attribute
    for demo_col in demographics.columns:
        if demo_col == 'age_years':  # Skip continuous version
            continue
    
        print(f"\n{'='*80}")
        print(f"Analyzing: {demo_col.upper()}")
        print(f"{'='*80}")
    
        sensitive_feature = demographics[demo_col]
    
        # Skip if too few categories or unknown
        if sensitive_feature.nunique() < 2 or 'Unknown' in sensitive_feature.values:
            print(f"  Skipping (insufficient data or unknown values)")
            continue
    
        # Group-level statistics
        group_stats = []
    
        for group in sensitive_feature.unique():
            mask = sensitive_feature == group
            n = mask.sum()
        
            if n < 30:  # Skip small groups
                continue
        
            # Performance metrics
            group_y_true = y_true[mask]
            group_y_pred_prob = y_pred_prob[mask]
            group_y_pred_binary = y_pred_binary[mask]
            group_decision = audit_df.loc[mask, 'decision']
        
            # Rates
            actual_default_rate = group_y_true.mean()
            predicted_high_risk_rate = group_y_pred_binary.mean()
            approve_rate = (group_decision == 'approve').mean()
            review_rate = (group_decision == 'review').mean()
            reject_rate = (group_decision == 'reject').mean()
        
            # Model performance
            if group_y_true.sum() > 0 and group_y_true.sum() < n:  # Need both classes
                group_auc = roc_auc_score(group_y_true, group_y_pred_prob)
            else:
                group_auc = np.nan
        
            group_stats.append({
                'demographic': demo_col,
                'group': group,
                'count': n,
                'actual_default_rate': actual_default_rate,
                'predicted_high_risk_rate': predicted_high_risk_rate,
                'approve_rate': approve_rate,
                'review_rate': review_rate,
                'reject_rate': reject_rate,
                'model_auc': group_auc
            })
        
            print(f"\n  {group} (n={n})")
            print(f"    Actual default: {actual_default_rate:.2%}")
            print(f"    Approve rate: {approve_rate:.2%}")
            print(f"    Review rate: {review_rate:.2%}")
            print(f"    Reject rate: {reject_rate:.2%}")
            if not np.isnan(group_auc):
                print(f"    Model AUC: {group_auc:.4f}")
    
        if len(group_stats) >= 2:
            group_df = pd.DataFrame(group_stats)
        
            # Calculate disparities
            max_approve = group_df['approve_rate'].max()
            min_approve = group_df['approve_rate'].min()
            approve_disparity = max_approve - min_approve
        
            max_reject = group_df['reject_rate'].max()
            min_reject = group_df['reject_rate'].min()
            reject_disparity = max_reject - min_reject
        
            print(f"\n  Disparity Analysis:")
            print(f"    Approve rate range: {min_approve:.2%} to {max_approve:.2%}")
            print(f"    Approve rate disparity: {approve_disparity:.2%}")
            print(f"    Reject rate range: {min_reject:.2%} to {max_reject:.2%}")
            print(f"    Reject rate disparity: {reject_disparity:.2%}")
        
            # Fairlearn metrics
            try:
                # Demographic Parity Difference
                dpd = demographic_parity_difference(
                    y_true, y_pred_binary, sensitive_features=sensitive_feature
                )
                print(f"    Demographic Parity Difference: {dpd:.4f}")
            
                # Equal Opportunity Difference (for qualified applicants - no default)
                qualified_mask = y_true == 0
                if qualified_mask.sum() > 0:
                    eod = equalized_odds_difference(
                        y_true, y_pred_binary, sensitive_features=sensitive_feature
                    )
                    print(f"    Equalized Odds Difference: {eod:.4f}")
                else:
                    eod = np.nan
            
                fairness_results.append({
                    'demographic': demo_col,
                    'num_groups': len(group_stats),
                    'approve_rate_min': min_approve,
                    'approve_rate_max': max_approve,
                    'approve_disparity': approve_disparity,
                    'reject_rate_min': min_reject,
                    'reject_rate_max': max_reject,
                    'reject_disparity': reject_disparity,
                    'demographic_parity_diff': dpd,
                    'equalized_odds_diff': eod
                })
            
            except Exception as e:
                print(f"    ⚠️  Error calculating fairness metrics: {str(e)[:50]}")
        
            # Save group-level stats
            group_df.to_csv(f"reports/fairness/{demo_col}_group_stats.csv", index=False)

    # =============================================================================
    # 5. OVERALL FAIRNESS SUMMARY
    # =============================================================================

    print("\n" + "=" * 80)
    print("OVERALL FAIRNESS SUMMARY")
    print("=" * 80)

    if fairness_results:
        fairness_df = pd.DataFrame(fairness_results)
    
        print("\nDisparity Summary:")
        print(fairness_df[['demographic', 'approve_disparity', 'reject_disparity', 
                            'demographic_parity_diff']].to_string(index=False))
    
        fairness_df.to_csv("reports/fairness/fairness_summary.csv", index=False)
        print("\nSaved: reports/fairness/fairness_summary.csv")
    
        # Visualizations
        print("\nGenerating visualizations...")
    
        # 1. Approve rate by demographic
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Approval Rates by Demographic Group", fontsize=16, y=1.00)
    
        plot_idx = 0
        for demo_col in demographics.columns:
            if demo_col == 'age_years':
                continue
        
            if os.path.exists(f"reports/fairness/{demo_col}_group_stats.csv"):
                group_data = pd.read_csv(f"reports/fairness/{demo_col}_group_stats.csv")
            
                if plot_idx < 4:
                    ax = axes[plot_idx // 2, plot_idx % 2]
                
                    # Bar plot
                    x_pos = np.arange(len(group_data))
                    bars = ax.bar(x_pos, group_data['approve_rate'] * 100)
                
                    # Color by rate (green = high approval, red = low)
                    colors = plt.cm.RdYlGn(group_data['approve_rate'])
                    for bar, color in zip(bars, colors):
                        bar.set_color(color)
                
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(group_data['group'], rotation=45, ha='right')
                    ax.set_ylabel('Approval Rate (%)')
                    ax.set_title(f'{demo_col.replace("_", " ").title()}')
                    ax.grid(axis='y', alpha=0.3)
                    ax.set_ylim(0, max(100, group_data['approve_rate'].max() * 110))
                
                    # Add value labels
                    for i, (_, row) in enumerate(group_data.iterrows()):
                        ax.text(i, row['approve_rate'] * 100 + 2, f"{row['approve_rate']*100:.1f}%",
                               ha='center', va='bottom', fontsize=9)
                
                    plot_idx += 1
    
        # Remove unused subplots
        for i in range(plot_idx, 4):
            fig.delaxes(axes[i // 2, i % 2])
    
        plt.tight_layout()
        plt.savefig("reports/fairness/approval_rates_by_demographic.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: reports/fairness/approval_rates_by_demographic.png")
    
        # 2. Decision distribution by demographic
        for demo_col in demographics.columns:
            if demo_col == 'age_years':
                continue
        
            if os.path.exists(f"reports/fairness/{demo_col}_group_stats.csv"):
                group_data = pd.read_csv(f"reports/fairness/{demo_col}_group_stats.csv")
            
                fig, ax = plt.subplots(figsize=(10, 6))
            
                x = np.arange(len(group_data))
                width = 0.25
            
                bars1 = ax.bar(x - width, group_data['approve_rate'] * 100, width, label='Approve', color='green', alpha=0.7)
                bars2 = ax.bar(x, group_data['review_rate'] * 100, width, label='Review', color='orange', alpha=0.7)
                bars3 = ax.bar(x + width, group_data['reject_rate'] * 100, width, label='Reject', color='red', alpha=0.7)
            
                ax.set_xlabel('Group')
                ax.set_ylabel('Percentage (%)')
                ax.set_title(f'Decision Distribution by {demo_col.replace("_", " ").title()}')
                ax.set_xticks(x)
                ax.set_xticklabels(group_data['group'], rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
            
                plt.tight_layout()
                plt.savefig(f"reports/fairness/decision_distribution_{demo_col}.png", dpi=150, bbox_inches='tight')
                plt.close()
    
        print("Saved: decision distribution plots")

    # =============================================================================
    # 6. GENERATE FAIRNESS REPORT
    # =============================================================================

    print("\n" + "=" * 80)
    print("GENERATING FAIRNESS REPORT")
    print("=" * 80)

    report = f"""# Fairness and Bias Audit Report

    **Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
    **Test Samples:** {len(audit_df):,}  
    **Overall Approve Rate:** {(audit_df['decision'] == 'approve').mean():.2%}  
    **Overall Reject Rate:** {(audit_df['decision'] == 'reject').mean():.2%}  

    ---

    ## Executive Summary

    This report analyzes demographic disparities in model predictions and lending decisions.

    **Key Findings:**
    """

    if fairness_results:
        max_disparity_row = fairness_df.loc[fairness_df['approve_disparity'].idxmax()]
        report += f"""
    - **Largest approval disparity:** {max_disparity_row['demographic']} ({max_disparity_row['approve_disparity']:.1%} difference between groups)
    - **Average demographic parity difference:** {fairness_df['demographic_parity_diff'].abs().mean():.4f}
    """

    report += """

    ---

    ## Methodology

    ### Metrics Used

    **Demographic Parity Difference (DPD)**
    - Measures difference in positive prediction rates between groups
    - Range: [-1, 1], with 0 indicating perfect parity
    - Interpretation: |DPD| < 0.1 is generally considered acceptable

    **Equalized Odds Difference (EOD)**
    - Measures difference in true positive and false positive rates
    - Ensures similar error rates across groups
    - Range: [-1, 1], with 0 indicating perfect equality

    **Group-Level Metrics**
    - Approval, Review, and Reject rates by demographic group
    - Actual default rates by group (ground truth)
    - Model performance (AUC) by group

    ---

    ## Results by Demographic

    """

    for demo_col in demographics.columns:
        if demo_col == 'age_years':
            continue
    
        if os.path.exists(f"reports/fairness/{demo_col}_group_stats.csv"):
            group_data = pd.read_csv(f"reports/fairness/{demo_col}_group_stats.csv")
        
            report += f"""
    ### {demo_col.replace('_', ' ').title()}

    """
        
            # Table
            report += "| Group | Count | Approve | Review | Reject | Actual Default |\n"
            report += "|---|---|---|---|---|---|\n"
        
            for _, row in group_data.iterrows():
                report += f"| {row['group']} | {row['count']:,} | {row['approve_rate']:.1%} | {row['review_rate']:.1%} | {row['reject_rate']:.1%} | {row['actual_default_rate']:.1%} |\n"
        
            # Disparity analysis
            if len(group_data) >= 2:
                max_approve = group_data['approve_rate'].max()
                min_approve = group_data['approve_rate'].min()
                disparity = max_approve - min_approve
            
                report += f"\n**Disparity:** {disparity:.1%} difference in approval rates\n"
            
                if disparity < 0.05:
                    report += "✅ **Assessment:** Low disparity (< 5%)\n"
                elif disparity < 0.10:
                    report += "⚠️ **Assessment:** Moderate disparity (5-10%)\n"
                else:
                    report += "🔴 **Assessment:** High disparity (> 10%)\n"
        
            report += "\n"

    report += """
    ---

    ## Gender Monitoring

    **Note:** Gender features were **excluded from the model** per ethical policy (Phase 1, Decision D-3).

    Gender is monitored for outcome disparities but **does not influence predictions**.

    """

    if 'gender' in demographics.columns and os.path.exists("reports/fairness/gender_group_stats.csv"):
        gender_data = pd.read_csv("reports/fairness/gender_group_stats.csv")
    
        report += "\n| Gender | Count | Approve | Review | Reject |\n"
        report += "|---|---|---|---|---|\n"
    
        for _, row in gender_data.iterrows():
            report += f"| {row['group']} | {row['count']:,} | {row['approve_rate']:.1%} | {row['review_rate']:.1%} | {row['reject_rate']:.1%} |\n"
    
        if len(gender_data) >= 2:
            max_approve = gender_data['approve_rate'].max()
            min_approve = gender_data['approve_rate'].min()
            disparity = max_approve - min_approve
        
            report += f"\n**Gender outcome disparity:** {disparity:.1%}\n"
            report += f"\n*This disparity exists despite gender not being in the model, suggesting correlated factors (income, employment, credit history) may differ by gender.*\n"

    report += """

    ---

    ## Interpretations and Recommendations

    ### Age

    Age is **included in the model** as a legally relevant factor (guardian requirements for minors).

    **Observed Patterns:**
    - Younger applicants (<25) may face higher rejection due to limited credit history
    - Middle-aged applicants (35-45) typically have highest approval rates (established income/credit)
    - Older applicants (55+) may see lower approval if nearing retirement

    **Recommendation:** Age disparities are expected and legally justified. Monitor for extreme disparities.

    ### Family Status

    Family status features (CNT_CHILDREN, CNT_FAM_MEMBERS) are **included** as they affect repayment capacity.

    **Observed Patterns:**
    - Larger families may need higher income to maintain same credit ratios
    - Single applicants may have more disposable income but less stability

    **Recommendation:** Monitor for disproportionate impact on families with children.

    ### Gender (Monitoring Only)

    Gender is **excluded from the model** but monitored for outcome disparities.

    **If disparities exist:**
    - Likely due to correlated factors (income gaps, employment differences)
    - Consider additional fairness interventions (e.g., income adjustment factors)
    - Document and report openly

    ---

    ## Fairness Assessment

    """

    if fairness_results:
        # Overall fairness verdict
        avg_dpd = fairness_df['demographic_parity_diff'].abs().mean()
        max_disparity = fairness_df['approve_disparity'].max()
    
        if max_disparity < 0.05 and avg_dpd < 0.1:
            verdict = "✅ **FAIR**"
            explanation = "The model shows low disparities across demographic groups (< 5% approval difference, DPD < 0.1)."
        elif max_disparity < 0.10 and avg_dpd < 0.15:
            verdict = "⚠️ **ACCEPTABLE WITH MONITORING**"
            explanation = "The model shows moderate disparities (5-10% approval difference). Continued monitoring and mitigation efforts recommended."
        else:
            verdict = "🔴 **REQUIRES INTERVENTION**"
            explanation = "The model shows significant disparities (> 10% approval difference). Fairness interventions needed."
    
        report += f"{verdict}\n\n{explanation}\n"

    report += """

    ---

    ## Limitations

    1. **Small Sample Sizes:** Some demographic subgroups have limited samples, reducing statistical confidence
    2. **Intersectionality:** This analysis examines one demographic at a time, not intersections (e.g., young women)
    3. **Unmeasured Factors:** Disparities may reflect unmeasured legitimate factors (wealth, assets, debt-to-income)
    4. **Static Analysis:** Real-world lending involves dynamic factors not captured in historical data

    ---

    ## Next Steps (Phase 4)

    - Integrate fairness monitoring into live decision dashboard
    - Alert officers when decisions create demographic disparities
    - Provide officers with fairness context for manual reviews
    - Document fairness considerations in PDF reports
    - Periodic re-audit (quarterly) as new data accumulates

    ---

    **Report Complete**
    """

    with open("reports/fairness/fairness_audit_report.md", "w") as f:
        f.write(report)

    print("\nSaved: reports/fairness/fairness_audit_report.md")

    print("\n" + "=" * 80)
    print("FAIRNESS AUDIT COMPLETE")
    print("=" * 80)
