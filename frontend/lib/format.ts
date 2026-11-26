import type { ApplicationStatus, Band, Lever } from "./types";

const inr = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });

export const money = (x: number | null | undefined) => (x == null ? "-" : inr.format(x));
export const percent = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;
export const date = (iso: string) =>
  new Date(iso).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });

export const BAND_TEXT: Record<Band, string> = {
  APPROVE: "Approved",
  REVIEW: "Referred to a loan officer",
  REJECT: "Not approved at this stage",
};

export const STATUS_TEXT: Record<ApplicationStatus, string> = {
  submitted: "Submitted",
  under_review: "Under review",
  approved: "Approved",
  rejected: "Not approved",
};

export const LEVER_TEXT: Record<Lever, string> = {
  credit: "Smaller loan",
  income: "Higher verified income",
  tenure: "Longer repayment",
  wait: "Re-apply later",
};

export const OUTCOME_TEXT: Record<Band, string> = {
  APPROVE: "Eligible for automatic approval",
  REVIEW: "Moves to officer review",
  REJECT: "No change",
};

/** Fallback when the backend sends no label: "AMT_CREDIT" -> "Amt credit". */
export const humanize = (feature: string) => {
  const s = feature.replace(/_/g, " ").toLowerCase();
  return s.charAt(0).toUpperCase() + s.slice(1);
};
