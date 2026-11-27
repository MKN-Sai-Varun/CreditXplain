# Accessibility Audit — CreditXplain frontend

The audit uses Lighthouse, which is built into Chrome's developer tools, so nothing needs installing. Each page takes about a minute.

## How to run it

1. Start the backend and frontend as in the README. Use a production build (`npm run build && npm run start`) so the scores are not affected by development mode.
2. Open each page below in Chrome and sign in where needed.
3. Open the developer tools (`Cmd+Option+I`), go to the **Lighthouse** tab, tick **Accessibility** and **Best practices** only, choose **Desktop**, and click **Analyze page load**.
4. Write the scores in the table. For each failed accessibility check, note its name, such as "Background and foreground colors do not have a sufficient contrast ratio".

## Results

Date: ________ · Chrome version: ________

| Page | Path | Accessibility | Best practices | Failed checks |
|---|---|---|---|---|
| Sign in | `/login` | | | |
| Register | `/register` | | | |
| Applications list | `/applications` | | | |
| New application | `/applications/new` | | | |
| Decision (Reject case) | `/applications/<id>` | | | |
| Officer queue | `/officer` | | | |
| Officer review | `/officer/applications/<id>` | | | |
| Analytics | `/officer/analytics` | | | |
| Fairness | `/officer/fairness` | | | |

## Keyboard check (5 minutes)

Using only the keyboard (Tab, Shift+Tab, Enter, Space):

| Check | Pass? |
|---|---|
| The whole application form can be filled in and submitted | |
| Recourse preferences can be changed and "What if…?" run | |
| The PDF report link can be reached and opened | |
| The focused element is always visible | |

Failed checks are fixed in the frontend and the pages re-audited. The paper reports the scores before and after.
