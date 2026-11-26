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

export interface ApplicationCreate {
  features: Features;
  tenure_months: number;
  recourse_preferences?: RecoursePreferences | null;
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
  created_at: string;
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
