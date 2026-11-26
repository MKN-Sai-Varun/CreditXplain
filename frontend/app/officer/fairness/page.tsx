"use client";

import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { RequireAuth } from "@/components/RequireAuth";
import { Alert, Button, Card, PageTitle, Spinner } from "@/components/ui";
import { errorMessage, fairnessAPI } from "@/lib/api";
import { percent } from "@/lib/format";
import type { DemographicStats, FairnessMetrics } from "@/lib/types";

const THRESHOLD = 0.1;
const AGE_ORDER = ["<25", "25-35", "35-45", "45-55", "55+"];

interface State {
  metrics: FairnessMetrics;
  age: DemographicStats[];
  family: DemographicStats[];
}

const loadAll = () =>
  Promise.all([fairnessAPI.metrics(), fairnessAPI.byAge(), fairnessAPI.byFamily()]).then(
    ([metrics, age, family]): State => ({
      metrics,
      age: [...age].sort((a, b) => AGE_ORDER.indexOf(a.group) - AGE_ORDER.indexOf(b.group)),
      family,
    }),
  );

function Gap({ label, value }: { label: string; value: number | null }) {
  const over = value != null && value > THRESHOLD;
  return (
    <Card>
      <p className="text-sm text-slate-500">{label}</p>
      <p className={`mt-1 text-2xl font-semibold ${over ? "text-red-700" : "text-slate-900"}`}>
        {value == null ? "-" : value.toFixed(3)}
      </p>
      <p className="text-xs text-slate-500">
        Largest gap in automatic approval rate between groups; alert above {THRESHOLD}.
      </p>
    </Card>
  );
}

function GroupChart({ title, rows }: { title: string; rows: DemographicStats[] }) {
  const data = rows.filter((r) => r.count > 0).map((r) => ({ ...r, ratePct: Math.round(r.approval_rate * 1000) / 10 }));
  return (
    <Card title={title}>
      {data.length === 0 ? (
        <p className="text-sm text-slate-600">No data yet.</p>
      ) : (
        <>
          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="group" tick={{ fontSize: 12 }} />
                <YAxis unit="%" domain={[0, 100]} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v) => [`${v}%`, "Approved automatically"]} />
                <Bar dataKey="ratePct" fill="#334155" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <table className="mt-3 w-full text-sm">
            <thead className="text-left text-slate-500">
              <tr>
                <th className="pb-1 font-medium">Group</th>
                <th className="pb-1 text-right font-medium">Applications</th>
                <th className="pb-1 text-right font-medium">Approved</th>
                <th className="pb-1 text-right font-medium">Not approved*</th>
                <th className="pb-1 text-right font-medium">Rate</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {data.map((r) => (
                <tr key={r.group}>
                  <td className="py-1">{r.group}</td>
                  <td className="py-1 text-right">{r.count}</td>
                  <td className="py-1 text-right">{r.approved}</td>
                  <td className="py-1 text-right">{r.rejected}</td>
                  <td className="py-1 text-right">{percent(r.approval_rate)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </Card>
  );
}

function Fairness() {
  const [state, setState] = useState<State | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    loadAll().then(setState).catch((err) => setError(errorMessage(err)));
  }, []);

  async function recalculate() {
    setBusy(true);
    setError(null);
    try {
      const res = await fairnessAPI.calculate();
      setNotice(res.message);
      if (res.success) setState(await loadAll());
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  if (!state && !error) return <Spinner />;
  const m = state?.metrics;

  return (
    <>
      <PageTitle
        title="Fairness monitoring"
        subtitle={
          m?.calculated_at
            ? `Last calculated ${new Date(m.calculated_at).toLocaleString("en-IN")}, from the model's decisions on all scored applications.`
            : "Not calculated yet. At least 50 scored applications are needed."
        }
        action={
          <Button onClick={recalculate} disabled={busy}>
            {busy ? "Calculating…" : "Recalculate"}
          </Button>
        }
      />
      {error && (
        <div className="mb-4">
          <Alert>{error}</Alert>
        </div>
      )}
      {notice && (
        <div className="mb-4">
          <Alert kind="info">{notice}</Alert>
        </div>
      )}
      {m?.has_alert && (
        <div className="mb-4">
          <Alert>
            <span className="font-medium">Fairness alert:</span> {m.alert_message}
          </Alert>
        </div>
      )}
      {state && (
        <div className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2">
            <Gap label="Age bands: demographic parity difference" value={m?.age_dpd ?? null} />
            <Gap label="Children / no children: demographic parity difference" value={m?.children_dpd ?? null} />
          </div>
          <div className="grid gap-6 lg:grid-cols-2">
            <GroupChart title="Automatic approval rate by age band" rows={state.age} />
            <GroupChart title="Automatic approval rate: children / no children" rows={state.family} />
          </div>
          <p className="text-xs text-slate-500">
            * Not approved counts every application the model did not approve automatically, so it includes cases referred
            for review. Equalised odds is not shown here: live applications have no repayment outcome yet, so it is
            reported only in the offline audit on historical data.
          </p>
        </div>
      )}
    </>
  );
}

export default function FairnessPage() {
  return (
    <RequireAuth roles={["officer", "admin"]}>
      <Fairness />
    </RequireAuth>
  );
}
