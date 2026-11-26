"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { DecisionSummary } from "../application/DecisionSummary";
import { Documents } from "../application/Documents";
import { LoanTermsCard } from "../application/LoanTermsCard";
import { Recourse } from "../application/Recourse";
import { RequireAuth } from "../RequireAuth";
import { Alert, Button, Card, PageTitle, Spinner, StatusBadge } from "../ui";
import { ActionPanel } from "./ActionPanel";
import { AuditTrail } from "./AuditTrail";
import { ExplanationTables } from "./ExplanationTables";
import { applicationAPI, errorMessage, officerAPI } from "@/lib/api";
import { LEVER_TEXT, date } from "@/lib/format";
import type { ApplicationDetail, AuditEntry } from "@/lib/types";

const both = (id: number) => Promise.all([applicationAPI.detail(id), officerAPI.auditLog(id)]);

function Review({ id }: { id: number }) {
  const [app, setApp] = useState<ApplicationDetail | null>(null);
  const [audit, setAudit] = useState<AuditEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [version, setVersion] = useState(0);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    both(id)
      .then(([a, log]) => {
        if (cancelled) return;
        setApp(a);
        setAudit(log);
      })
      .catch((err) => !cancelled && setError(errorMessage(err)));
    return () => {
      cancelled = true;
    };
  }, [id]);

  async function refresh(message: string) {
    setNotice(message);
    try {
      const [a, log] = await both(id);
      setApp(a);
      setAudit(log);
      setVersion((v) => v + 1);   // remount panels that keep their own copy
    } catch (err) {
      setError(errorMessage(err));
    }
  }

  async function compliance() {
    setBusy(true);
    try {
      await officerAPI.downloadCompliance(id);
      setAudit(await officerAPI.auditLog(id));   // the download is logged
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  if (error && !app) return <Alert>{error}</Alert>;
  if (!app) return <Spinner />;
  const d = app.decision;
  const prefs = app.recourse_preferences;

  return (
    <>
      <PageTitle
        title={`Review application #${app.id}`}
        subtitle={`Applicant #${app.user_id} · submitted ${date(app.created_at)}`}
        action={
          <div className="flex items-center gap-3">
            <StatusBadge status={app.status} />
            {d && (
              <Button variant="secondary" onClick={compliance} disabled={busy}>
                {busy ? "Preparing…" : "Compliance report (PDF)"}
              </Button>
            )}
          </div>
        }
      />
      {notice && (
        <div className="mb-4">
          <Alert kind="success">{notice}</Alert>
        </div>
      )}
      {error && (
        <div className="mb-4">
          <Alert>{error}</Alert>
        </div>
      )}
      {!d ? (
        <Card>
          <p className="text-sm text-slate-600">This application has not been scored yet.</p>
        </Card>
      ) : (
        <div className="space-y-6">
          <DecisionSummary app={app} />
          <ActionPanel app={app} onDone={refresh} />
          <ExplanationTables explanation={app.explanation} />
          {d.decision !== "APPROVE" && (
            <>
              <Card title="Applicant's recourse limits">
                <p className="text-sm text-slate-700">
                  {prefs
                    ? `Cannot use: ${prefs.locked_levers.map((l) => LEVER_TEXT[l].toLowerCase()).join(", ") || "nothing locked"} · ` +
                      `loan reduction up to ${Math.round(prefs.max_loan_reduction * 100)}% · wait up to ${prefs.max_wait_months} months`
                    : "None set: all levers allowed."}
                </p>
              </Card>
              <Recourse key={`recourse-${version}`} appId={app.id} initial={app.explanation?.counterfactuals ?? []} initialPrefs={prefs} />
            </>
          )}
          {app.loan_terms && <LoanTermsCard terms={app.loan_terms} />}
          <Documents
            key={`docs-${version}`}
            appId={app.id}
            initialDocuments={app.documents}
            requests={app.document_requests}
            canUpload={false}
          />
          <AuditTrail entries={audit} />
        </div>
      )}
      <p className="mt-6 text-sm">
        <Link href="/officer" className="text-slate-700 underline">
          Back to the review queue
        </Link>
      </p>
    </>
  );
}

export function OfficerReview() {
  const params = useParams<{ id: string }>();
  const id = Number(params.id);
  return (
    <RequireAuth roles={["officer", "admin"]}>
      {Number.isInteger(id) && id > 0 ? <Review id={id} /> : <Alert>Unknown application.</Alert>}
    </RequireAuth>
  );
}
