import { Card } from "../ui";
import type { AuditEntry } from "@/lib/types";

const words = (s: string | null) => (s ? s.replace(/_/g, " ") : "-");

export function AuditTrail({ entries }: { entries: AuditEntry[] }) {
  return (
    <Card title="Audit trail">
      {entries.length === 0 ? (
        <p className="text-sm text-slate-600">No entries yet.</p>
      ) : (
        <ol className="space-y-3 text-sm">
          {entries.map((e) => (
            <li key={e.id} className="border-l-2 border-slate-200 pl-3">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <span className="font-medium capitalize text-slate-900">{words(e.action)}</span>
                <span className="text-xs text-slate-500">{new Date(e.timestamp).toLocaleString("en-IN")}</span>
              </div>
              <div className="text-xs text-slate-500">
                {e.officer_id ? `Officer #${e.officer_id}` : "Applicant"} · {words(e.previous_status)} → {words(e.new_status)}
              </div>
              {e.justification && <p className="mt-1 text-slate-700">{e.justification}</p>}
            </li>
          ))}
        </ol>
      )}
    </Card>
  );
}
