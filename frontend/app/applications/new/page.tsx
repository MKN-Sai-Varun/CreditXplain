"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { PreferencesEditor } from "@/components/application/PreferencesEditor";
import { RequireAuth } from "@/components/RequireAuth";
import { Alert, Button, Card, Field, Input, PageTitle, Select } from "@/components/ui";
import { applicationAPI, errorMessage } from "@/lib/api";
import {
  EDUCATION,
  EMPTY,
  FAMILY_STATUS,
  HOUSING,
  INCOME_TYPE,
  RELATIONSHIPS,
  SAMPLES,
  TENURES,
  estimateYearlyRepayment,
  isMinor,
  toFeatures,
  toGuardian,
  validate,
  type FormValues,
} from "@/lib/applicationForm";
import { money } from "@/lib/format";
import type { RecoursePreferences } from "@/lib/types";

const DEFAULT_PREFS: RecoursePreferences = { locked_levers: [], max_loan_reduction: 0.5, max_wait_months: 12 };

function NewApplicationForm() {
  const router = useRouter();
  const [v, setV] = useState<FormValues>(EMPTY);
  const [prefs, setPrefs] = useState<RecoursePreferences>(DEFAULT_PREFS);
  const [showPrefs, setShowPrefs] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const [busy, setBusy] = useState<string | null>(null);

  const set = <K extends keyof FormValues>(key: K, value: FormValues[K]) => setV((prev) => ({ ...prev, [key]: value }));
  const text = (key: keyof FormValues) => ({
    value: v[key] as string,
    onChange: (e: React.ChangeEvent<HTMLInputElement>) => set(key, e.target.value as never),
  });

  const minor = isMinor(v);
  const estimate = Number(v.loanAmount) > 0 ? estimateYearlyRepayment(Number(v.loanAmount), v.tenureMonths) : null;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const problems = validate(v);
    setErrors(problems);
    if (problems.length) return;

    setBusy("Submitting application");
    try {
      const app = await applicationAPI.create({
        features: toFeatures(v),
        tenure_months: v.tenureMonths,
        recourse_preferences: showPrefs ? prefs : null,
        guardian: toGuardian(v),
      });
      setBusy("Assessing application");
      await applicationAPI.predict(app.id);
      router.push(`/applications/${app.id}`);
    } catch (err) {
      setErrors([errorMessage(err)]);
      setBusy(null);
    }
  }

  return (
    <>
      <PageTitle
        title="New application"
        subtitle="Amounts are in rupees. Income and repayment are yearly amounts."
        action={
          <div className="flex flex-wrap gap-2">
            {Object.entries(SAMPLES).map(([key, s]) => (
              <Button key={key} type="button" variant="secondary" onClick={() => setV(s.values)}>
                Fill: {s.label}
              </Button>
            ))}
          </div>
        }
      />
      <form onSubmit={submit} className="space-y-6">
        <Card title="Loan">
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Loan amount" hint="How much you want to borrow">
              <Input type="number" min={1} inputMode="numeric" required {...text("loanAmount")} />
            </Field>
            <Field label="Price of the goods" hint="For example the price of the car or appliance">
              <Input type="number" min={1} inputMode="numeric" required {...text("goodsPrice")} />
            </Field>
            <Field label="Repayment period">
              <Select
                options={TENURES.map((t) => `${t} months`)}
                value={`${v.tenureMonths} months`}
                onChange={(e) => set("tenureMonths", Number.parseInt(e.target.value, 10))}
              />
            </Field>
            <Field
              label="Yearly repayment"
              hint={estimate ? `About ${money(estimate)} a year at 12% over ${v.tenureMonths} months` : "Total repaid per year"}
            >
              <div className="flex gap-2">
                <Input type="number" min={1} inputMode="numeric" required {...text("yearlyRepayment")} />
                <Button type="button" variant="secondary" disabled={!estimate} onClick={() => estimate && set("yearlyRepayment", String(estimate))}>
                  Use estimate
                </Button>
              </div>
            </Field>
          </div>
        </Card>

        <Card title="About you">
          <div className="grid gap-4 sm:grid-cols-3">
            <Field label="Yearly income" hint="Enter 0 if you have no income">
              <Input type="number" min={0} inputMode="numeric" required {...text("yearlyIncome")} />
            </Field>
            <Field label="Age">
              <Input type="number" min={1} max={100} required {...text("age")} />
            </Field>
            <Field label="Years in current job" hint="Leave empty if not employed">
              <Input type="number" min={0} step="0.5" {...text("yearsEmployed")} />
            </Field>
            <Field label="Children">
              <Input type="number" min={0} {...text("children")} />
            </Field>
            <Field label="Family members" hint="Including you">
              <Input type="number" min={1} {...text("familyMembers")} />
            </Field>
            <Field label="Education">
              <Select options={EDUCATION} value={v.education} onChange={(e) => set("education", e.target.value)} />
            </Field>
            <Field label="Income type">
              <Select options={INCOME_TYPE} value={v.incomeType} onChange={(e) => set("incomeType", e.target.value)} />
            </Field>
            <Field label="Family status">
              <Select options={FAMILY_STATUS} value={v.familyStatus} onChange={(e) => set("familyStatus", e.target.value)} />
            </Field>
            <Field label="Housing">
              <Select options={HOUSING} value={v.housing} onChange={(e) => set("housing", e.target.value)} />
            </Field>
          </div>
          <div className="mt-4 flex flex-wrap gap-6 text-sm">
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={v.ownsCar} onChange={(e) => set("ownsCar", e.target.checked)} /> I own a car
            </label>
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={v.ownsProperty} onChange={(e) => set("ownsProperty", e.target.checked)} /> I own property
            </label>
          </div>
        </Card>

        {minor && (
          <Card title="Guardian">
            <p className="mb-3 text-sm text-slate-600">
              Applicants under 18 apply together with a guardian. The application is assessed on the guardian&apos;s
              income and employment, and a loan officer makes the decision. The officer will need the guardian&apos;s ID
              proof and income proof, which you can upload on the decision page.
            </p>
            <div className="grid gap-4 sm:grid-cols-3">
              <Field label="Guardian's full name">
                <Input required {...text("guardianName")} />
              </Field>
              <Field label="Relationship">
                <Select
                  options={RELATIONSHIPS.map((r) => r.label)}
                  value={RELATIONSHIPS.find((r) => r.value === v.guardianRelationship)?.label}
                  onChange={(e) =>
                    set("guardianRelationship", RELATIONSHIPS.find((r) => r.label === e.target.value)?.value ?? "other")
                  }
                />
              </Field>
              <Field label="Guardian's age">
                <Input type="number" min={18} max={100} required {...text("guardianAge")} />
              </Field>
              <Field label="Guardian's yearly income">
                <Input type="number" min={0} inputMode="numeric" required {...text("guardianIncome")} />
              </Field>
              <Field label="Guardian's income type">
                <Select options={INCOME_TYPE} value={v.guardianIncomeType} onChange={(e) => set("guardianIncomeType", e.target.value)} />
              </Field>
              <Field label="Guardian's years in current job" hint="Leave empty if not employed">
                <Input type="number" min={0} step="0.5" {...text("guardianYearsEmployed")} />
              </Field>
            </div>
          </Card>
        )}

        <Card title="Credit-bureau scores (optional)">
          <p className="mb-3 text-sm text-slate-600">
            In a real bank these come from credit bureaus. For this demo you can enter values between 0 and 1;
            empty fields use typical values.
          </p>
          <div className="grid gap-4 sm:grid-cols-3">
            {(["bureau1", "bureau2", "bureau3"] as const).map((key, i) => (
              <Field key={key} label={`Score ${i + 1}`}>
                <Input type="number" min={0} max={1} step="0.01" {...text(key)} />
              </Field>
            ))}
          </div>
        </Card>

        <Card
          title="If the application is not approved"
          action={
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={showPrefs} onChange={(e) => setShowPrefs(e.target.checked)} /> Set my limits
            </label>
          }
        >
          <p className="text-sm text-slate-600">
            We suggest changes that could improve the decision. Tell us what you cannot change, so the suggestions fit
            you. You can also change this later on the decision page.
          </p>
          {showPrefs && (
            <div className="mt-4">
              <PreferencesEditor value={prefs} onChange={setPrefs} />
            </div>
          )}
        </Card>

        {errors.length > 0 && (
          <Alert>
            <ul className="list-disc pl-5">
              {errors.map((e) => (
                <li key={e}>{e}</li>
              ))}
            </ul>
          </Alert>
        )}
        <div className="flex justify-end">
          <Button type="submit" disabled={busy != null}>
            {busy ? `${busy}…` : "Submit and see decision"}
          </Button>
        </div>
      </form>
    </>
  );
}

export default function NewApplicationPage() {
  return (
    <RequireAuth roles={["applicant"]}>
      <NewApplicationForm />
    </RequireAuth>
  );
}
