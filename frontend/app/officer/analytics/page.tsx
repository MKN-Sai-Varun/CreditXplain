"use client";

import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { RequireAuth } from "@/components/RequireAuth";
import { Alert, Card, PageTitle, Spinner } from "@/components/ui";
import { errorMessage, officerAPI } from "@/lib/api";
import { percent } from "@/lib/format";
import type { ApprovalRateStats } from "@/lib/types";

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <p className="text-sm text-slate-500">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-slate-900">{value}</p>
    </Card>
  );
}

function Analytics() {
  const [stats, setStats] = useState<ApprovalRateStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    officerAPI.approvalRates().then(setStats).catch((err) => setError(errorMessage(err)));
  }, []);

  if (error) return <Alert>{error}</Alert>;
  if (!stats) return <Spinner />;

  const other = stats.total_applications - stats.approved - stats.rejected - stats.under_review;
  const data = [
    { name: "Approved", count: stats.approved, color: "#059669" },
    { name: "Under review", count: stats.under_review, color: "#d97706" },
    { name: "Not approved", count: stats.rejected, color: "#dc2626" },
    ...(other > 0 ? [{ name: "Not yet scored", count: other, color: "#94a3b8" }] : []),
  ];

  return (
    <>
      <PageTitle title="Analytics" subtitle="Current status of all applications, including officer decisions." />
      <div className="mb-6 grid gap-4 sm:grid-cols-4">
        <Stat label="Applications" value={String(stats.total_applications)} />
        <Stat label="Approved" value={String(stats.approved)} />
        <Stat label="Under review" value={String(stats.under_review)} />
        <Stat label="Approval rate" value={percent(stats.approval_rate)} />
      </div>
      <Card title="Applications by status">
        {stats.total_applications === 0 ? (
          <p className="text-sm text-slate-600">No applications yet.</p>
        ) : (
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" name="Applications" radius={[4, 4, 0, 0]}>
                  {data.map((d) => (
                    <Cell key={d.name} fill={d.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </Card>
    </>
  );
}

export default function AnalyticsPage() {
  return (
    <RequireAuth roles={["officer", "admin"]}>
      <Analytics />
    </RequireAuth>
  );
}
