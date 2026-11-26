"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { RequireAuth } from "@/components/RequireAuth";
import { Alert, Card, PageTitle, Spinner, StatusBadge } from "@/components/ui";
import { applicationAPI, errorMessage } from "@/lib/api";
import { date } from "@/lib/format";
import type { ApplicationSummary } from "@/lib/types";

function ApplicationList() {
  const [apps, setApps] = useState<ApplicationSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    applicationAPI.list().then(setApps).catch((err) => setError(errorMessage(err)));
  }, []);

  return (
    <>
      <PageTitle
        title="My applications"
        subtitle="Open an application to see its decision, the reasons behind it and what could change it."
        action={
          <Link href="/applications/new" className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700">
            New application
          </Link>
        }
      />
      {error && <Alert>{error}</Alert>}
      {!apps && !error && <Spinner />}
      {apps && apps.length === 0 && (
        <Card>
          <p className="text-sm text-slate-600">You have no applications yet.</p>
        </Card>
      )}
      {apps && apps.length > 0 && (
        <ul className="divide-y divide-slate-200 overflow-hidden rounded-lg border border-slate-200 bg-white">
          {apps.map((a) => (
            <li key={a.id}>
              <Link href={`/applications/${a.id}`} className="flex items-center justify-between px-5 py-4 hover:bg-slate-50">
                <div>
                  <p className="font-medium text-slate-900">Application #{a.id}</p>
                  <p className="text-sm text-slate-500">Submitted {date(a.created_at)}</p>
                </div>
                <StatusBadge status={a.status} />
              </Link>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

export default function ApplicationsPage() {
  return (
    <RequireAuth roles={["applicant"]}>
      <ApplicationList />
    </RequireAuth>
  );
}
