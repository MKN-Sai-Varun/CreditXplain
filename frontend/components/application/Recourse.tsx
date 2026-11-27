"use client";

import { useState } from "react";

import { Alert, Button, Card, Spinner } from "../ui";
import { PreferencesEditor } from "./PreferencesEditor";
import { applicationAPI, errorMessage } from "@/lib/api";
import { LEVER_TEXT, OUTCOME_TEXT, humanize, money, percent } from "@/lib/format";
import type { RecourseChange, RecourseOption, RecoursePreferences } from "@/lib/types";

const MONEY_FIELDS = new Set(["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY"]);

function ChangeLine({ ch }: { ch: RecourseChange }) {
  if (ch.feature === "WAIT_MONTHS")
    return (
      <li>
        Re-apply after <strong>{ch.recommended} months</strong>, staying in the same job and making no new credit
        applications meanwhile
      </li>
    );
  if (!MONEY_FIELDS.has(ch.feature)) return null;   // time effects of waiting are listed in the PDF report
  return (
    <li>
      {ch.label}: {money(ch.original)} → <strong>{money(ch.recommended)}</strong>
    </li>
  );
}

function RetrainBadge({ passed }: { passed?: boolean | null }) {
  if (passed == null) return null;
  return passed ? (
    <span className="mt-1 inline-block rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-medium text-emerald-800">
      Likely to hold if the model is updated
    </span>
  ) : (
    <span className="mt-1 inline-block rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600">
      May change if the model is updated
    </span>
  );
}

function OptionCard({ opt, index }: { opt: RecourseOption; index: number }) {
  return (
    <div className="rounded-md border border-slate-200 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="font-medium text-slate-900">
          Option {index + 1}: {opt.levers.map((l) => LEVER_TEXT[l]).join(" + ")}
        </p>
        <span className="text-sm font-medium text-emerald-700">{OUTCOME_TEXT[opt.new_decision]}</span>
      </div>
      <RetrainBadge passed={opt.retrain_check_passed} />
      <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-700">
        {opt.changes.map((ch) => (
          <ChangeLine key={ch.feature} ch={ch} />
        ))}
      </ul>
      <p className="mt-2 text-xs text-slate-500">
        New estimated risk {percent(opt.new_probability)}
        {opt.drivers && opt.drivers.length > 0 && (
          <> · Why it helps: {opt.drivers.map((d) => d.label || humanize(d.feature)).join("; ")}</>
        )}
      </p>
    </div>
  );
}

export function Recourse({
  appId,
  initial,
  initialPrefs,
}: {
  appId: number;
  initial: RecourseOption[];
  initialPrefs: RecoursePreferences | null | undefined;
}) {
  const [options, setOptions] = useState(initial);
  const [prefs, setPrefs] = useState<RecoursePreferences>(
    initialPrefs ?? { locked_levers: [], max_loan_reduction: 0.5, max_wait_months: 12 },
  );
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recomputed, setRecomputed] = useState(false);

  async function recompute() {
    setBusy(true);
    setError(null);
    try {
      setOptions(await applicationAPI.recourse(appId, prefs));
      setRecomputed(true);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card
      title="What could change the decision"
      action={
        <Button variant="secondary" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
          What if…?
        </Button>
      }
    >
      <p className="mb-4 text-sm text-slate-600">
        Each option would move the application to the next stage according to the model. These are model-based
        estimates, not guarantees.
      </p>
      {open && (
        <div className="mb-4 rounded-md bg-slate-50 p-4">
          <PreferencesEditor value={prefs} onChange={setPrefs} />
          <div className="mt-4 flex justify-end">
            <Button onClick={recompute} disabled={busy}>
              {busy ? "Finding options…" : "Show options for these limits"}
            </Button>
          </div>
        </div>
      )}
      {error && <Alert>{error}</Alert>}
      {busy ? (
        <Spinner label="Finding options" />
      ) : options.length === 0 ? (
        <p className="text-sm text-slate-600">
          {recomputed
            ? "No option fits these limits. Try allowing a larger change or a longer wait."
            : "No change within the supported limits (loan amount, income, repayment period, waiting up to 12 months) would improve this decision."}
        </p>
      ) : (
        <div className="space-y-3">
          {options.map((opt, i) => (
            <OptionCard key={opt.levers.join("-") + i} opt={opt} index={i} />
          ))}
        </div>
      )}
    </Card>
  );
}
