"use client";

import { useState } from "react";

import { Alert, Button, Card } from "../ui";
import { errorMessage, officerAPI } from "@/lib/api";
import type { ApplicationDetail, OfficerActionType } from "@/lib/types";

const LABEL: Record<OfficerActionType, string> = {
  approve: "Approve",
  reject: "Reject",
  request_documents: "Request documents",
};

/** Mirrors the backend rule: overriding the model or requesting documents needs a justification. */
export function needsJustification(app: ApplicationDetail, action: OfficerActionType) {
  const model = app.decision?.decision;
  return (
    action === "request_documents" ||
    (action === "approve" && model !== "APPROVE") ||
    (action === "reject" && model === "APPROVE")
  );
}

export function ActionPanel({ app, onDone }: { app: ApplicationDetail; onDone: (message: string) => void }) {
  const [action, setAction] = useState<OfficerActionType>("request_documents");
  const [justification, setJustification] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const required = needsJustification(app, action);
  const missing = required && justification.trim() === "";

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (missing) return;
    setBusy(true);
    setError(null);
    try {
      const res = await officerAPI.act(app.id, action, justification.trim() || undefined);
      setJustification("");
      onDone(`${LABEL[action]}: done. Status is now ${res.application.status.replace("_", " ")}.`);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card title="Officer action">
      <form onSubmit={submit} className="space-y-4">
        <fieldset className="flex flex-wrap gap-4 text-sm">
          <legend className="sr-only">Action</legend>
          {(Object.keys(LABEL) as OfficerActionType[]).map((a) => (
            <label key={a} className="flex items-center gap-2">
              <input type="radio" name="action" value={a} checked={action === a} onChange={() => setAction(a)} />
              {LABEL[a]}
            </label>
          ))}
        </fieldset>
        <label className="block text-sm">
          <span className="font-medium text-slate-800">
            {action === "request_documents" ? "Documents needed (shown to the applicant)" : "Justification"}
            {required ? " (required)" : " (optional)"}
          </span>
          <textarea
            value={justification}
            onChange={(e) => setJustification(e.target.value)}
            rows={3}
            className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none focus:ring-1 focus:ring-slate-500"
          />
          {required && action !== "request_documents" && (
            <span className="mt-1 block text-xs text-amber-700">This goes against the model&apos;s decision, so a reason is needed.</span>
          )}
        </label>
        {error && <Alert>{error}</Alert>}
        <div className="flex justify-end">
          <Button type="submit" disabled={busy || missing}>
            {busy ? "Saving…" : LABEL[action]}
          </Button>
        </div>
      </form>
    </Card>
  );
}
