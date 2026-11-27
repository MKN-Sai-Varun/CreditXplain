# CreditXplain Frontend

Next.js 16 (App Router, React 19, TypeScript, Tailwind CSS 4) frontend for the CreditXplain API.

## Status

| Area | State |
|---|---|
| Sign in, register, session restore | Done (Phase 4.9) |
| Applicant portal: applications list, new application, decision page | Done (Phase 4.9) |
| Officer dashboard: review queue, application review and actions, analytics, fairness monitoring | Done (Phase 4.10) |

## Run it

Prerequisites: Node.js 20+, and the backend running on http://localhost:8000 (see `backend/README.md`).

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000. To point at another backend, create `frontend/.env.local` with:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Checks:

```bash
npm run lint
npm run build      # also generates the route types used by the next check
npx tsc --noEmit
```

## Pages

| Route | Who | What |
|---|---|---|
| `/` | everyone | Landing page; signed-in users are sent to their area |
| `/login`, `/register` | everyone | Sign in / create an applicant account |
| `/applications` | applicant | Own applications with status |
| `/applications/new` | applicant | Application form (sample profiles, repayment estimate, optional bureau scores, recourse limits; a guardian section appears for applicants under 18); submitting scores the application |
| `/applications/[id]` | applicant (own), officer | Decision and risk, main reasons, recourse options with "What if…?", loan terms, document requests and upload, PDF report |
| `/officer` | officer | Review queue: Review-band applications, oldest first, with risk, amount, tenure, documents and waiting time |
| `/officer/applications/[id]` | officer | Decision, officer action (approve / reject / request documents with justification), SHAP and LIME tables, recourse and the applicant's limits, loan terms, documents (read-only), audit trail, compliance report |
| `/officer/analytics` | officer | Status totals, approval rate and a chart of applications by status |
| `/officer/fairness` | officer | Age-band and children/no-children parity gaps, alert banner, approval rate per group (chart and table), recalculation |

## Structure

```
app/                     routes (App Router)
components/              AuthProvider, RequireAuth, AppHeader, AuthForm, ui
components/application/  decision page parts: DecisionSummary, Recourse, PreferencesEditor, Documents, LoanTermsCard, ApplicationView
components/officer/      ActionPanel, ExplanationTables, AuditTrail, OfficerReview
lib/api.ts               typed API client (axios), token storage, error messages, downloads
lib/types.ts             response types matching backend/schemas.py
lib/applicationForm.ts   form options, sample profiles, form values -> model features
lib/format.ts            money, percentages, dates, labels
```

## Notes

- The login token is kept in `localStorage` and sent as a Bearer token.
- `cacheComponents` is enabled in `next.config.ts`, so the dynamic `/applications/[id]` page wraps its client view in `<Suspense>`.
- Plain-language names for model features come from the backend (`label` on explanation rows), the same wording as the PDF reports.
- Gender is not collected: the model excludes it.
- Demo accounts: `python3 -m backend.seed_db` (from the project root) creates `applicant@creditxplain.com` and `officer@creditxplain.com`, both with password `password123`. Officer accounts cannot be self-registered.
