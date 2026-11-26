# CreditXplain Frontend

Next.js 16 (App Router, React 19, TypeScript, Tailwind CSS 4) frontend for the CreditXplain API.

## Status

| Area | State |
|---|---|
| Sign in, register, session restore | Done (Phase 4.9) |
| Applicant portal: applications list, new application, decision page | Done (Phase 4.9) |
| Officer dashboard | Phase 4.10 (placeholder page at `/officer`) |

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
npx tsc --noEmit
npm run build
```

## Pages

| Route | Who | What |
|---|---|---|
| `/` | everyone | Landing page; signed-in users are sent to their area |
| `/login`, `/register` | everyone | Sign in / create an applicant account |
| `/applications` | applicant | Own applications with status |
| `/applications/new` | applicant | Application form (sample profiles, repayment estimate, optional bureau scores, recourse limits); submitting scores the application |
| `/applications/[id]` | applicant (own), officer | Decision and risk, main reasons, recourse options with "What if…?", loan terms, document requests and upload, PDF report |
| `/officer` | officer | Placeholder until Phase 4.10 |

## Structure

```
app/                     routes (App Router)
components/              AuthProvider, RequireAuth, AppHeader, AuthForm, ui
components/application/  decision page parts: DecisionSummary, Recourse, PreferencesEditor, Documents, ApplicationView
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
