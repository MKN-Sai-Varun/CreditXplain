import { Card, DecisionBadge, StatusBadge } from "../ui";
import { humanize, money, percent } from "@/lib/format";
import type { ApplicationDetail, ShapValue } from "@/lib/types";

const STATUS_FOR = { APPROVE: "approved", REVIEW: "under_review", REJECT: "rejected" } as const;

export function DecisionSummary({ app }: { app: ApplicationDetail }) {
  const d = app.decision!;
  const changedByOfficer = app.status !== STATUS_FOR[d.decision];
  return (
    <Card>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-sm text-slate-500">Decision</p>
          <div className="mt-1 flex items-center gap-2">
            <DecisionBadge band={d.decision} />
            {changedByOfficer && (
              <span className="flex items-center gap-1 text-sm text-slate-600">
                now <StatusBadge status={app.status} /> after officer review
              </span>
            )}
          </div>
        </div>
        <div className="text-right">
          <p className="text-sm text-slate-500">Estimated chance of repayment difficulty</p>
          <p className="text-2xl font-semibold text-slate-900">{percent(d.probability)}</p>
        </div>
      </div>
      <dl className="mt-4 grid grid-cols-2 gap-4 border-t border-slate-100 pt-4 text-sm sm:grid-cols-3">
        <div>
          <dt className="text-slate-500">Loan amount</dt>
          <dd className="font-medium">{money(Number(app.features.AMT_CREDIT))}</dd>
        </div>
        <div>
          <dt className="text-slate-500">Repayment period</dt>
          <dd className="font-medium">{app.tenure_months} months</dd>
        </div>
        <div>
          <dt className="text-slate-500">Automatic approval below</dt>
          <dd className="font-medium">{percent(d.approve_threshold)} risk</dd>
        </div>
      </dl>
    </Card>
  );
}

export function Reasons({ rows }: { rows: ShapValue[] }) {
  const top = rows.slice(0, 5);
  const max = Math.max(...top.map((r) => Math.abs(r.value)), 1e-9);
  return (
    <Card title="Main reasons">
      {top.length === 0 ? (
        <p className="text-sm text-slate-600">No explanation is available for this application.</p>
      ) : (
        <>
          <p className="mb-3 text-sm text-slate-600">The factors that weighed most in this decision, strongest first.</p>
          <ul className="space-y-3">
            {top.map((r) => {
              const raises = r.value > 0;
              return (
                <li key={r.feature}>
                  <div className="flex items-baseline justify-between gap-4 text-sm">
                    <span className="text-slate-800">{r.label || humanize(r.feature)}</span>
                    <span className={raises ? "text-red-700" : "text-emerald-700"}>{raises ? "raises risk" : "lowers risk"}</span>
                  </div>
                  <div className="mt-1 h-1.5 rounded-full bg-slate-100">
                    <div
                      className={`h-1.5 rounded-full ${raises ? "bg-red-400" : "bg-emerald-500"}`}
                      style={{ width: `${(Math.abs(r.value) / max) * 100}%` }}
                    />
                  </div>
                </li>
              );
            })}
          </ul>
        </>
      )}
    </Card>
  );
}
