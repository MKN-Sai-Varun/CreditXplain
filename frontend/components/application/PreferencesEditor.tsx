"use client";

import { Field, Select } from "../ui";
import { LEVERS, WAITS } from "@/lib/applicationForm";
import { LEVER_TEXT } from "@/lib/format";
import type { Lever, RecoursePreferences } from "@/lib/types";

const CANNOT: Record<Lever, string> = {
  credit: "I cannot borrow less",
  income: "I cannot increase my verified income",
  tenure: "I cannot take a longer repayment period",
  wait: "I cannot wait before re-applying",
};

export function PreferencesEditor({
  value,
  onChange,
}: {
  value: RecoursePreferences;
  onChange: (next: RecoursePreferences) => void;
}) {
  const toggle = (lever: Lever, locked: boolean) =>
    onChange({
      ...value,
      locked_levers: locked ? [...value.locked_levers, lever] : value.locked_levers.filter((l) => l !== lever),
    });

  return (
    <div className="space-y-4">
      <fieldset className="grid gap-2 sm:grid-cols-2">
        <legend className="mb-1 text-sm font-medium text-slate-800">What can you not change?</legend>
        {LEVERS.map((lever) => (
          <label key={lever} className="flex items-center gap-2 text-sm text-slate-700">
            <input
              type="checkbox"
              checked={value.locked_levers.includes(lever)}
              onChange={(e) => toggle(lever, e.target.checked)}
            />
            {CANNOT[lever]}
            <span className="sr-only">({LEVER_TEXT[lever]})</span>
          </label>
        ))}
      </fieldset>
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Largest loan reduction I would accept">
          <Select
            options={["5%", "10%", "20%", "30%", "40%", "50%"]}
            value={`${Math.round(value.max_loan_reduction * 100)}%`}
            disabled={value.locked_levers.includes("credit")}
            onChange={(e) => onChange({ ...value, max_loan_reduction: Number.parseInt(e.target.value, 10) / 100 })}
          />
        </Field>
        <Field label="Longest I would wait before re-applying">
          <Select
            options={WAITS.map((w) => (w === 0 ? "No waiting" : `${w} months`))}
            value={value.max_wait_months === 0 ? "No waiting" : `${value.max_wait_months} months`}
            disabled={value.locked_levers.includes("wait")}
            onChange={(e) =>
              onChange({ ...value, max_wait_months: e.target.value === "No waiting" ? 0 : Number.parseInt(e.target.value, 10) })
            }
          />
        </Field>
      </div>
    </div>
  );
}
