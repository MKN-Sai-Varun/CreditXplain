import type { Features, Lever } from "./types";

// Category values as the model knows them. The backend turns
// "Secondary / secondary special" into the one-hot column
// NAME_EDUCATION_TYPE_Secondary_secondary_special.
export const EDUCATION = [
  "Academic degree",
  "Higher education",
  "Incomplete higher",
  "Secondary / secondary special",
  "Lower secondary",
];
export const INCOME_TYPE = [
  "Working",
  "Commercial associate",
  "State servant",
  "Businessman",
  "Pensioner",
  "Student",
  "Unemployed",
  "Maternity leave",
];
export const FAMILY_STATUS = ["Married", "Civil marriage", "Single / not married", "Separated", "Widow"];
export const HOUSING = [
  "House / apartment",
  "Rented apartment",
  "With parents",
  "Municipal apartment",
  "Office apartment",
  "Co-op apartment",
];
export const TENURES = [12, 24, 36, 48, 60];
export const WAITS = [0, 3, 6, 12];
export const LEVERS: Lever[] = ["credit", "income", "tenure", "wait"];

export interface FormValues {
  loanAmount: string;
  goodsPrice: string;
  yearlyIncome: string;
  yearlyRepayment: string;
  tenureMonths: number;
  age: string;
  yearsEmployed: string;
  children: string;
  familyMembers: string;
  education: string;
  incomeType: string;
  familyStatus: string;
  housing: string;
  ownsCar: boolean;
  ownsProperty: boolean;
  bureau1: string;
  bureau2: string;
  bureau3: string;
}

export const EMPTY: FormValues = {
  loanAmount: "",
  goodsPrice: "",
  yearlyIncome: "",
  yearlyRepayment: "",
  tenureMonths: 24,
  age: "",
  yearsEmployed: "",
  children: "0",
  familyMembers: "2",
  education: "Higher education",
  incomeType: "Working",
  familyStatus: "Married",
  housing: "House / apartment",
  ownsCar: false,
  ownsProperty: true,
  bureau1: "",
  bureau2: "",
  bureau3: "",
};

// Demo profiles: one lands in each decision band with the current model.
export const SAMPLES: Record<string, { label: string; values: FormValues }> = {
  strong: {
    label: "Strong profile",
    values: {
      ...EMPTY, loanAmount: "450000", goodsPrice: "450000", yearlyIncome: "180000", yearlyRepayment: "24000",
      age: "35", yearsEmployed: "5", children: "0", familyMembers: "2", ownsCar: true,
      bureau1: "0.65", bureau2: "0.70", bureau3: "0.65",
    },
  },
  middle: {
    label: "Borderline profile",
    values: {
      ...EMPTY, loanAmount: "540000", goodsPrice: "450000", yearlyIncome: "157500", yearlyRepayment: "27000",
      age: "38", yearsEmployed: "4", children: "1", familyMembers: "3", education: "Secondary / secondary special",
      bureau1: "0.58", bureau2: "0.58", bureau3: "0.58",
    },
  },
  weak: {
    label: "Higher-risk profile",
    values: {
      ...EMPTY, loanAmount: "900000", goodsPrice: "750000", yearlyIncome: "112500", yearlyRepayment: "45000",
      age: "28", yearsEmployed: "1", children: "0", familyMembers: "1", education: "Secondary / secondary special",
      familyStatus: "Single / not married", ownsProperty: false,
      bureau1: "0.35", bureau2: "0.40", bureau3: "0.35",
    },
  },
};

/** Yearly repayment of an amortising loan at an assumed 12% a year, for pre-filling the form. */
export function estimateYearlyRepayment(amount: number, months: number, annualRate = 0.12) {
  const r = annualRate / 12;
  const emi = (amount * r * (1 + r) ** months) / ((1 + r) ** months - 1);
  return Math.round(emi * 12);
}

const num = (s: string) => (s.trim() === "" ? null : Number(s));
const DAYS_PER_YEAR = 365.25;

/** Form values -> raw features the backend scores. Empty optional fields are left out (defaults apply). */
export function toFeatures(v: FormValues): Features {
  const f: Features = {
    AMT_CREDIT: Number(v.loanAmount),
    AMT_GOODS_PRICE: Number(v.goodsPrice),
    AMT_INCOME_TOTAL: Number(v.yearlyIncome),
    AMT_ANNUITY: Number(v.yearlyRepayment),
    CNT_CHILDREN: Number(v.children),
    CNT_FAM_MEMBERS: Number(v.familyMembers),
    NAME_EDUCATION_TYPE: v.education,
    NAME_INCOME_TYPE: v.incomeType,
    NAME_FAMILY_STATUS: v.familyStatus,
    NAME_HOUSING_TYPE: v.housing,
    FLAG_OWN_CAR: v.ownsCar ? "Y" : "N",
    FLAG_OWN_REALTY: v.ownsProperty ? "Y" : "N",
  };
  const age = num(v.age);
  if (age != null) f.DAYS_BIRTH = Math.round(-age * DAYS_PER_YEAR);
  const employed = num(v.yearsEmployed);
  if (employed != null) f.DAYS_EMPLOYED = Math.round(-employed * DAYS_PER_YEAR);
  for (const [key, value] of [["EXT_SOURCE_1", v.bureau1], ["EXT_SOURCE_2", v.bureau2], ["EXT_SOURCE_3", v.bureau3]]) {
    const n = num(value);
    if (n != null) f[key] = n;
  }
  return f;
}

/** Problems to show before submitting, or an empty list. */
export function validate(v: FormValues): string[] {
  const errors: string[] = [];
  const positive = (s: string, name: string) => {
    if (!(Number(s) > 0)) errors.push(`${name} must be a positive number`);
  };
  positive(v.loanAmount, "Loan amount");
  positive(v.goodsPrice, "Goods price");
  positive(v.yearlyIncome, "Yearly income");
  positive(v.yearlyRepayment, "Yearly repayment");
  const age = num(v.age);
  if (age == null || age < 18 || age > 100) errors.push("Age must be between 18 and 100");
  const employed = num(v.yearsEmployed);
  if (employed != null && (employed < 0 || (age != null && employed > age - 14)))
    errors.push("Years in current job is out of range");
  for (const [s, name] of [[v.bureau1, "Bureau score 1"], [v.bureau2, "Bureau score 2"], [v.bureau3, "Bureau score 3"]]) {
    const n = num(s);
    if (n != null && (n < 0 || n > 1)) errors.push(`${name} must be between 0 and 1`);
  }
  return errors;
}
