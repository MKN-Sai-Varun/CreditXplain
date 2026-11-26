import { Card } from "../ui";
import { money, percent } from "@/lib/format";
import type { LoanTerms } from "@/lib/types";

export function LoanTermsCard({ terms }: { terms: LoanTerms }) {
  const rows: [string, string][] = [
    ["Loan amount", money(terms.loan_amount)],
    ["Interest rate (yearly)", percent(terms.interest_rate, 2)],
    ["Repayment period", `${terms.tenure_months} months`],
    ["Monthly installment (EMI)", money(terms.monthly_emi)],
    ["Total payable", money(terms.total_payable)],
  ];
  return (
    <Card title="Loan terms">
      <dl className="grid gap-3 text-sm sm:grid-cols-2">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-4 border-b border-slate-100 pb-2">
            <dt className="text-slate-500">{k}</dt>
            <dd className="font-medium">{v}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-3 text-xs text-slate-500">Interest rates in this academic system are illustrative and are not an offer.</p>
    </Card>
  );
}
