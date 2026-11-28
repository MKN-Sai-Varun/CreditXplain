# Manuscript (IEEE Access)

The paper's LaTeX source. Tables, figures and references are generated into `reports/paper/` by
`python3 -m src.paper_figures`; this folder holds the text.

| File | Contents | Status |
|---|---|---|
| `main.tex` | Title page, authors, abstract, keywords, section list, references, biographies | Abstract, keywords, affiliation e-mails and bios still to fill in |
| `sections/introduction.tex` | I. Introduction | Draft (batch 1) |
| `sections/related_work.tex` | II. Related Work | Draft (batch 1) |
| `sections/method.tex` | III. System and Method | Draft (batch 2) |
| `sections/setup.tex` | IV. Experimental Setup | Draft (batch 2) |
| `sections/results.tex` | V. Results | Batch 3 |
| `sections/discussion.tex` | VI. Discussion and Limitations | Batch 3 |
| `sections/conclusion.tex` | VII. Conclusion | Batch 3 |

## Overleaf project layout

```
main.tex            <- this folder's main.tex
ieeeaccess.cls      <- from the official template, unchanged
references.bib      <- reports/paper/references.bib
sections/*.tex      <- this folder's sections
figures/*.pdf       <- reports/paper/figures/*.pdf
tables/*.tex        <- reports/paper/tables/*.tex
a1.png              <- template sample photo, until the authors' photos are added
```

Everything marked `FILL IN` in `main.tex` needs the authors' details.
