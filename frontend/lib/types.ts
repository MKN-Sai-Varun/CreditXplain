// Shapes returned by the FastAPI backend (backend/schemas.py)

export type Role = "applicant" | "officer" | "admin";
export type Band = "APPROVE" | "REVIEW" | "REJECT";
export type ApplicationStatus = "submitted" | "under_review" | "approved" | "rejected";
export type Lever = "credit" | "income" | "tenure" | "wait";

export interface User {
  id: number;
  email: string;
  full_name: string | null;
  role: Role;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface RecoursePreferences {
  locked_levers: Lever[];
  max_loan_reduction: number;
  max_wait_months: number;
}

export type Features = Record<string, string | number | null>;

export type GuardianRelationship = "parent" | "legal_guardian" | "other";

/** Guardian applying with an applicant under 18 */
export interface Guardian {
  full_name: string;
  relationship: GuardianRelationship;
  age: number;
  yearly_income: number;
  income_type: string;
  years_employed?: number | null;
}

export interface ApplicationCreate {
  features: Features;
  tenure_months: number;
  recourse_preferences?: RecoursePreferences | null;
  guardian?: Guardian | null;
}

export interface ApplicationSummary {
  id: number;
  user_id: number;
  status: ApplicationStatus;
  created_at: string;
  updated_at: string;
}

export interface Decision {
  id: number;
  application_id: number;
  decision: Band;
  probability: number;
  approve_threshold: number;
  reject_threshold: number;
  eligibility?: Eligibility | null;
  created_at: string;
}

/** Rules checked before the model (backend/services/eligibility_service.py) */
export interface Eligibility {
  rules: { code: "minor" | "no_income"; reason: string }[];
  assessed_on: "applicant" | "guardian";
  automatic_approval: boolean;
  required_documents: string[];
  model_decision?: Band;
}

export interface ShapValue {
  feature: string;
  value: number;
  feature_value: number | string | null;
  label?: string | null;
}

export interface LimeFeature {
  feature: string;
  weight: number;
  rule?: string | null;
  label?: string | null;
}

export interface RecourseChange {
  feature: string;
  label: string;
  lever: Lever;
  original: number;
  recommended: number;
  display_original?: number;
  display_recommended?: number;
}

export interface Driver {
  feature: string;
  delta: number;
  label?: string | null;
}

export interface RecourseOption {
  levers: Lever[];
  changes: RecourseChange[];
  num_changes: number;
  wait_months: number;
  original_probability?: number | null;
  new_probability: number;
  new_decision: Band;
  reaches_approve: boolean;
  cost?: number | null;
  robust_margin?: number | null;
  plausible?: boolean | null;
  drivers?: Driver[] | null;
  /** true: also reached the better band on all retrained models; null: not checked */
  retrain_check_passed?: boolean | null;
  ensemble_agreement?: string | null;
}

export interface Explanation {
  shap_values: ShapValue[];
  lime_explanation: LimeFeature[];
  counterfactuals?: RecourseOption[] | null;
}

export interface LoanTerms {
  loan_amount: number;
  interest_rate: number;
  tenure_months: number;
  monthly_emi: number;
  total_payable: number;
}

export interface DocumentInfo {
  id: number;
  application_id: number;
  doc_type: string;
  original_name: string;
  content_type: string;
  size_bytes: number;
  uploaded_at: string;
}

export interface DocumentRequest {
  message: string | null;
  requested_at: string;
}

export interface ApplicationDetail extends ApplicationSummary {
  features: Features;
  tenure_months: number;
  recourse_preferences?: RecoursePreferences | null;
  guardian?: Guardian | null;
  decision?: Decision | null;
  explanation?: Explanation | null;
  loan_terms?: LoanTerms | null;
  documents: DocumentInfo[];
  document_requests: DocumentRequest[];
}

export interface DocumentsOverview {
  requests: DocumentRequest[];
  documents: DocumentInfo[];
}

// Officer and fairness endpoints

export type OfficerActionType = "approve" | "reject" | "request_documents";

export interface OfficerActionResponse {
  message: string;
  application: ApplicationSummary;
  loan_terms?: LoanTerms | null;
}

export interface AuditEntry {
  id: number;
  application_id: number;
  officer_id: number | null;
  action: string;
  justification: string | null;
  previous_status: string | null;
  new_status: string | null;
  timestamp: string;
}

export interface ApprovalRateStats {
  total_applications: number;
  approved: number;
  rejected: number;
  under_review: number;
  approval_rate: number;
}

export interface FairnessMetrics {
  age_dpd: number | null;
  children_dpd: number | null;
  has_alert: boolean;
  alert_message: string | null;
  metrics_detail: Record<string, unknown>;
  calculated_at: string | null;
}

/** Per-group counts. `rejected` is count - approved, so it includes Review cases. */
export interface DemographicStats {
  group: string;
  count: number;
  approved: number;
  rejected: number;
  approval_rate: number;
}

export interface FairnessCalculation {
  message: string;
  success: boolean;
  metrics?: { age_dpd?: number; family_dpd?: number; has_alert?: boolean; total_applications?: number };
}
