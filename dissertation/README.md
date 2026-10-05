# Dissertation (CSCK700) — source and build

The dissertation is written as Markdown, one file per chapter, and built into
the University's Word template so that the source stays reviewable in git.

| Path | Content |
|---|---|
| `chapters/00_front.md` | Abstract, acknowledgements, AI-use statement |
| `chapters/01_…06_*.md` | Chapters 1–6 (template structure) |
| `chapters/90_references.md` | Harvard reference list (66 sources, 50 from 2023+) |
| `chapters/95_appendices.md` | Appendices A–E |
| `figures/` | Figures (copied from experiment results; `fig_architecture.png` from `make_architecture.py`) |
| `template/template.docx` | The Computing dissertation template (unmodified) |
| `verify_refs.py`, `ref_queries*.txt`, `refs_crossref*.json` | Reference metadata checked against Crossref |
| `build_docx.py` | Markdown → template (styles, captions with SEQ fields, appendices) |
| `update_fields.ps1` | Opens the result in Word, refreshes TOC / lists / numbering, exports PDF |
| `build/Abouabdou_Badr_Dissertation.docx` / `.pdf` | The built draft |

## Rebuild

```
.venv/Scripts/python dissertation/build_docx.py
powershell -ExecutionPolicy Bypass -File dissertation/update_fields.ps1
```

Appendices A and B are page images of the proposal and SDR, rendered from
`build/*.pdf` (exported from the original .docx files by Word), so their
headings do not join the dissertation's numbering.

## Before submission (author)

- Complete the bracketed placeholders: acknowledgements, AI-use statement
  (check the module policy), personal growth paragraph (6.2), code-access line
  (Appendix E).
- Sign the declaration page in Word.
- Every number was taken from experiment FINDINGS / results files and
  re-checked by `experiments/AUDIT/verify_results.py`; if text is edited,
  keep numbers consistent with those files.
- Body word count (Chapter 1 to the end of Chapter 6, incl. tables and
  captions): about 16,800 (limit 12,000–18,000).
