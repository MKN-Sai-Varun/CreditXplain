"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { RequireAuth } from "../RequireAuth";
import { Alert, Button, Card, PageTitle, Spinner, StatusBadge } from "../ui";
import { DecisionSummary, Reasons } from "./DecisionSummary";
import { Documents } from "./Documents";
import { LoanTermsCard } from "./LoanTermsCard";
import { Recourse } from "./Recourse";
import { applicationAPI, errorMessage } from "@/lib/api";
import { date } from "@/lib/format";
import type { ApplicationDetail } from "@/lib/types";

function Detail({ id }: { id: number }) {
  const [app, setApp] = useState<ApplicationDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    applicationAPI
      .detail(id)
      .then((a) => !cancelled && setApp(a))
      .catch((err) => !cancelled && setError(errorMessage(err)));
    return () => {
      cancelled = true;
    };
  }, [id]);

  async function reload() {
    try {
      setApp(await applicationAPI.detail(id));
    } catch (err) {
      setError(errorMessage(err));
    }
  }

  async function score() {
    setBusy("Assessing application");
    setError(null);
    try {
      await applicationAPI.predict(id);
      await reload();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  }

  async function report() {
    setBusy("Preparing report");
    try {
      await applicationAPI.downloadReport(id);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  }

  if (error && !app) return <Alert>{error}</Alert>;
  if (!app) return <Spinner />;

  const d = app.decision;
  const options = app.explanation?.counterfactuals ?? [];

  return (
    <>
      <PageTitle
        title={`Application #${app.id}`}
        subtitle={`Submitted ${date(app.created_at)}`}
        action={
          <div className="flex items-center gap-3">
            <StatusBadge status={app.status} />
            {d && (
              <Button variant="secondary" onClick={report} disabled={busy != null}>
                {busy === "Preparing report" ? "Preparing…" : "Download report (PDF)"}
              </Button>
            )}
          </div>
        }
      />
      {error && (
        <div className="mb-4">
          <Alert>{error}</Alert>
        </div>
      )}
      {!d ? (
        <Card>
          <p className="text-sm text-slate-600">This application has not been assessed yet.</p>
          <Button className="mt-3" onClick={score} disabled={busy != null}>
            {busy ? `${busy}…` : "Get decision"}
          </Button>
        </Card>
      ) : (
        <div className="space-y-6">
          <DecisionSummary app={app} />
          <Reasons rows={app.explanation?.shap_values ?? []} />
          {d.decision !== "APPROVE" && (
            <Recourse appId={app.id} initial={options} initialPrefs={app.recourse_preferences} />
          )}
          {app.loan_terms && <LoanTermsCard terms={app.loan_terms} />}
          <Documents appId={app.id} initialDocuments={app.documents} requests={app.document_requests} />
          <p className="text-xs text-slate-500">
            This is an automated credit-risk assessment. A loan officer reviews referred cases, and you can ask for any
            decision to be reviewed by a person.
          </p>
        </div>
      )}
      <p className="mt-6 text-sm">
        <Link href="/applications" className="text-slate-700 underline">
          Back to my applications
        </Link>
      </p>
    </>
  );
}

export function ApplicationView() {
  const params = useParams<{ id: string }>();
  const id = Number(params.id);
  return (
    <RequireAuth roles={["applicant", "officer", "admin"]}>
      {Number.isInteger(id) && id > 0 ? <Detail id={id} /> : <Alert>Unknown application.</Alert>}
    </RequireAuth>
  );
}
