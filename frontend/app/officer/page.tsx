"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { RequireAuth } from "@/components/RequireAuth";
import { Alert, Card, PageTitle, Spinner } from "@/components/ui";
import { errorMessage, officerAPI } from "@/lib/api";
import { date, money, percent } from "@/lib/format";
import type { ApplicationDetail } from "@/lib/types";

const daysSince = (iso: string, now: number) => Math.max(0, Math.floor((now - new Date(iso).getTime()) / 86_400_000));

function Queue() {
  const [apps, setApps] = useState<ApplicationDetail[] | null>(null);
  const [loadedAt, setLoadedAt] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    officerAPI
      .queue()
      .then((rows) => {
        setLoadedAt(Date.now());
        setApps(rows);
      })
      .catch((err) => setError(errorMessage(err)));
  }, []);

  return (
    <>
      <PageTitle title="Review queue" subtitle="Applications referred for officer review, oldest first." />
      {error && <Alert>{error}</Alert>}
      {!apps && !error && <Spinner />}
      {apps && apps.length === 0 && (
        <Card>
          <p className="text-sm text-slate-600">No applications are waiting for review.</p>
        </Card>
      )}
      {apps && apps.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-50 text-slate-600">
              <tr>
                <th className="px-4 py-3 font-medium">Application</th>
                <th className="px-4 py-3 font-medium">Model decision</th>
                <th className="px-4 py-3 font-medium">Risk</th>
                <th className="px-4 py-3 font-medium">Loan amount</th>
                <th className="px-4 py-3 font-medium">Tenure</th>
                <th className="px-4 py-3 font-medium">Documents</th>
                <th className="px-4 py-3 font-medium">Waiting</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {apps.map((a) => (
                <tr key={a.id} className="hover:bg-slate-50">
                  <td className="px-4 py-3">
                    <Link href={`/officer/applications/${a.id}`} className="font-medium text-slate-900 underline">
                      #{a.id}
                    </Link>
                    <div className="text-xs text-slate-500">{date(a.created_at)}</div>
                  </td>
                  <td className="px-4 py-3">{a.decision?.decision ?? "Not scored"}</td>
                  <td className="px-4 py-3">{a.decision ? percent(a.decision.probability) : "-"}</td>
                  <td className="px-4 py-3">{money(Number(a.features.AMT_CREDIT))}</td>
                  <td className="px-4 py-3">{a.tenure_months} months</td>
                  <td className="px-4 py-3">
                    {a.documents.length} uploaded
                    {a.document_requests.length > 0 && (
                      <div className="text-xs text-amber-700">{a.document_requests.length} requested</div>
                    )}
                  </td>
                  <td className="px-4 py-3">{daysSince(a.created_at, loadedAt)} days</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

export default function OfficerQueuePage() {
  return (
    <RequireAuth roles={["officer", "admin"]}>
      <Queue />
    </RequireAuth>
  );
}
