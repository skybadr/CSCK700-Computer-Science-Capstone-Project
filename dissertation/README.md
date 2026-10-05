# Dissertation (CSCK700) — sources and build tools

The dissertation was drafted as Markdown, one file per chapter, built into the
University of Liverpool Computing dissertation template, and then finalised in
Word. The submitted document is the Word file; the Markdown sources mirror its
content.

| Path | Content |
|---|---|
| `chapters/00_front.md` | Abstract and acknowledgements (including the generative-AI statement) |
| `chapters/01_…06_*.md` | Chapters 1–6, following the template structure |
| `chapters/90_references.md` | Harvard reference list (66 sources, 50 from 2023 or later) |
| `chapters/95_appendices.md` | Appendices A–E |
| `figures/` | Figures (from the experiment results; `fig_architecture.png` from `make_architecture.py`) |
| `template/template.docx` | The Computing dissertation template |
| `verify_refs.py`, `ref_queries*.txt`, `refs_crossref*.json` | Reference metadata checked against Crossref |
| `build_docx.py` | Builds the Markdown into the template (styles, auto-numbered captions, appendices) |
| `edit_docx_*.py` | In-place revisions applied to the Word file during finalisation |
| `update_fields.ps1` | Opens the document in Word, refreshes the contents and numbering, exports a PDF |

Appendix A is the approved proposal (`../BadrAbouabdou_Proposal_approved_by_DA_DL.docx`)
and Appendix B the Specification and Design Report, both included as page images.

Every number in the dissertation comes from the experiment results and is
re-checked by `../experiments/AUDIT/verify_results.py`.
