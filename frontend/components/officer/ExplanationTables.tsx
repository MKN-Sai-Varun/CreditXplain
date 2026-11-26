import { Card } from "../ui";
import { humanize } from "@/lib/format";
import type { Explanation } from "@/lib/types";

const fmt = (v: number | string | null) => (typeof v === "number" ? (Math.abs(v) >= 1000 ? v.toFixed(0) : v.toPrecision(4)) : v ?? "-");

export function ExplanationTables({ explanation }: { explanation: Explanation | null | undefined }) {
  const shap = explanation?.shap_values.slice(0, 10) ?? [];
  const lime = explanation?.lime_explanation.slice(0, 10) ?? [];
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card title="SHAP contributions (log-odds)">
        {shap.length === 0 ? (
          <p className="text-sm text-slate-600">Not available.</p>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-left text-slate-500">
              <tr>
                <th className="pb-2 font-medium">Feature</th>
                <th className="pb-2 font-medium">Value</th>
                <th className="pb-2 text-right font-medium">Contribution</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {shap.map((r) => (
                <tr key={r.feature}>
                  <td className="py-1.5 pr-2">
                    {r.label || humanize(r.feature)}
                    <div className="font-mono text-xs text-slate-400">{r.feature}</div>
                  </td>
                  <td className="py-1.5 pr-2 font-mono text-xs">{fmt(r.feature_value)}</td>
                  <td className={`py-1.5 text-right font-mono ${r.value > 0 ? "text-red-700" : "text-emerald-700"}`}>
                    {r.value > 0 ? "+" : ""}
                    {r.value.toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
      <Card title="LIME local rules (log-odds)">
        {lime.length === 0 ? (
          <p className="text-sm text-slate-600">Not available.</p>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-left text-slate-500">
              <tr>
                <th className="pb-2 font-medium">Rule</th>
                <th className="pb-2 text-right font-medium">Weight</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {lime.map((r) => (
                <tr key={r.feature}>
                  <td className="py-1.5 pr-2">
                    {r.label || humanize(r.feature)}
                    <div className="font-mono text-xs text-slate-400">{r.rule ?? r.feature}</div>
                  </td>
                  <td className={`py-1.5 text-right font-mono ${r.weight > 0 ? "text-red-700" : "text-emerald-700"}`}>
                    {r.weight > 0 ? "+" : ""}
                    {r.weight.toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="mt-3 text-xs text-slate-500">
          LIME explains the 25 features with the highest model gain, perturbed around this applicant.
        </p>
      </Card>
    </div>
  );
}
