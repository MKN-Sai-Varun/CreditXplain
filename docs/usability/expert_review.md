# Expert Review — CreditXplain

A structured review by one or two evaluators, ideally a faculty member and someone with banking or lending knowledge. It takes about 30–40 minutes per evaluator and needs no other participants.

The method is a **heuristic evaluation** (Nielsen's ten usability heuristics, with Nielsen's 0–4 severity scale) combined with a short **task walkthrough** and five questions specific to explanations and recourse. Evaluators work independently. Their findings are then merged into one list, fixed where possible, and reported as "N issues found, M fixed".

---

## 1. Setup (by the project lead, before the session)

1. Start the backend and frontend as in the README, and seed the demo users (`python3 -m backend.seed_db`).
2. Create a fresh applicant account for the evaluator. Keep the officer account ready (`officer@creditxplain.com`) for task 4.
3. Give the evaluator this document and a copy of the findings table (section 5). Explain only what the system is for: an automated credit decision with reasons and suggestions on what could change it. Do not demonstrate the app.
4. Record the evaluator only by a code (E1, E2) and their background in one phrase (e.g. "faculty, computer science", "bank employee"). Collect no names.

**Consent wording (read out):**
> This review looks at how clear the CreditXplain application is. You will use the application with made-up applicant data and note any problems you find. No personal data is collected, and you are identified only by a code. You can stop at any time. Your notes will be summarised, without your name, in the project report and a research paper.

---

## 2. Tasks (about 15 minutes)

Do the tasks in order. For each, note whether you could complete it without help, and any point where you hesitated or were unsure.

| # | Task | Completed? (yes / with difficulty / no) |
|---|---|---|
| 1 | Sign in as the applicant. Submit a new application using the **Higher-risk profile**. Find out what the decision was and the **main reason** for it. | |
| 2 | On the same application, pick the option you would choose to improve the decision. Say **exactly what you would have to change** and what the decision would become. | |
| 3 | Suppose you **cannot increase your income**. Use "What if…?" to find options that still work. Then download the PDF report and check it says the same thing. | |
| 4 | Submit a second application with the **Strong profile** and find the **monthly installment and interest rate**. | |
| 5 | Sign in as the officer. Open an application **under review**, decide on it, and give a reason. Open the compliance report. | |

---

## 3. Heuristics (about 10 minutes)

For each heuristic, write down any problem you saw during the tasks. Rate each problem on the severity scale below. One heuristic may have several problems, or none.

| # | Heuristic | What to look for in this application |
|---|---|---|
| H1 | Visibility of system status | Is it always clear what stage the application is at (submitted, decided, under review)? |
| H2 | Match between system and the real world | Are reasons and options in everyday words, with amounts in rupees and time in months? |
| H3 | User control and freedom | Can the applicant change preferences and go back without losing work? |
| H4 | Consistency and standards | Do the web page and the PDF use the same words for the same things? |
| H5 | Error prevention | Does the form stop impossible values (e.g. negative income) before submitting? |
| H6 | Recognition rather than recall | Is everything needed to understand an option visible next to it? |
| H7 | Flexibility and efficiency | Can an officer get to the cases that need a decision quickly? |
| H8 | Aesthetic and minimalist design | Is any information shown that an applicant does not need? |
| H9 | Help users recognise and recover from errors | Are error messages clear and do they say what to do? |
| H10 | Help and documentation | Is it explained what "likely to hold if the model is updated" means? |

**Severity scale (Nielsen):**
0 = not a usability problem · 1 = cosmetic, fix if time allows · 2 = minor · 3 = major, important to fix · 4 = catastrophic, must fix before release.

---

## 4. Explanation and recourse questions (about 5 minutes)

Answer each on a 1–5 scale (1 = strongly disagree, 5 = strongly agree), with a sentence of comment if you can.

| # | Statement | 1–5 | Comment |
|---|---|---|---|
| X1 | I understood why the application received its decision. | | |
| X2 | The suggested options were things an applicant could actually do. | | |
| X3 | It was clear which option asks for the least change. | | |
| X4 | The "likely to hold / may change if the model is updated" labels were understandable. | | |
| X5 | A loan officer could use the reasons shown to justify a decision to a customer. | | |

Open question: **What one change would make the explanations clearer?**

---

## 5. Findings table (one row per problem)

Fill in during or straight after the session. Copy rows as needed.

| Evaluator | Task or screen | Heuristic (H1–H10) or X1–X5 | Problem, in one sentence | Severity (0–4) |
|---|---|---|---|---|
| E1 | | | | |

---

## 6. After the sessions (project lead and developer)

1. Merge the evaluators' tables. Problems that are the same count once, at the higher severity.
2. Fix every problem of severity 3–4, and the cheap 1–2 ones. Record each fix in `docs/KNOWN_ISSUES.md` with the evaluator code.
3. Report in the paper: the number of evaluators and their backgrounds, the task completion table, the problems found by severity, how many were fixed, and the X1–X5 scores **as individual answers, not as averages**. A summary statistic means little with one or two evaluators.
4. State the limitation clearly: an expert review is not a user study, and a study with real applicants is future work.
