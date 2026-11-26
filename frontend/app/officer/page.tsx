"use client";

import { RequireAuth } from "@/components/RequireAuth";
import { Card, PageTitle } from "@/components/ui";

export default function OfficerHome() {
  return (
    <RequireAuth roles={["officer", "admin"]}>
      <PageTitle title="Officer dashboard" />
      <Card>
        <p className="text-sm text-slate-600">
          The review queue, application review and fairness monitoring are added in the next phase. The officer
          API is already available at <code className="rounded bg-slate-100 px-1">/api/officer</code>.
        </p>
      </Card>
    </RequireAuth>
  );
}
