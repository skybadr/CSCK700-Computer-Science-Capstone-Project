# Pre-writing audit (2026-10-05)

**Why:** before drafting the dissertation, every number it will cite was
recomputed from the raw data by an independent script that imports no
experiment code. Datasets, API logs, results tables and FINDINGS files were
cross-checked.

**Run:** `.venv/Scripts/python experiments/AUDIT/verify_results.py`
(output: `audit_output.txt`). Result: **117 / 117 checks pass.**

## What was checked

| Area | Checks |
|---|---|
| A. Datasets | sizes (500 / 1,000 / 400 / 456), unique ids and texts, category quotas, 800/200 stratified split, 150 synthetic; exam vs pilot / v2 / probe: zero exact and zero 12-word passage overlap; v2 test vs pilot and probe: zero 12-word overlap |
| B. Exp 10 raw data | 13,438 responses, unique keys, one model snapshot, 79 truncated, 1 empty; total cost $2.49 recomputed from token usage; every results row re-joined to its stored response (text and token counts identical); split / category columns match the frozen dataset; TCR reproduced with cl100k |
| C. Exp 10 findings | noise ceiling (29.6% identical, median 0.930, 31.8% < 0.85, τ quantile 0.640); RQ1 differences, p-values and ranking flip; ceiling-normalised category fragility; cost table (input saving and total cost change); test labels, APCS 1.0.0 and baseline accuracies, McNemar p; RQ2 OLS R² and unique R², morphology n.s. |
| D. Exp 11 findings | 3,828 responses, $0.61; cost_usd formula; selectors re-applied from the frozen JSON files; accuracy, cost change and below-τ rate per policy; H1–H3 discordant counts and Holm p-values; QA correctness; protected-compressor cost table |
| E. Supporting | Exp 09a/09b summary values; pilot Exp 01, 04, 06, 07 headline values |
| F. Docs and artefact | shipped `rules_default.json` = documented APCS 1.0.0 and = exam selector; 11 unit tests pass; experiment log table well-formed; README + FINDINGS per experiment |

## Issues found and fixed

1. **Experiment log, Exp 11 row:** stray text (`| 05e | 2026-09-29 |.61. |`)
   had been pasted into the row, breaking the table. Fixed; the row now
   ends "Exam API cost $0.61."
2. **Experiment log, 05e row:** `|ρ|` broke the Markdown table (pipe
   characters). Reworded as "absolute ρ".
3. **Experiment log, QA correctness range:** said "32–53% compressed"; the
   measured range is 21–53% (random deletion 21–31%). Corrected to match
   Exp 11 FINDINGS.

No data, result or conclusion changed.

## Notes for the write-up (not errors)

- **Scope of ceiling numbers:** the 29.6% / 0.930 / 31.8% figures are on the
  800 dev prompts (all 1,456 prompts give 27.3% / 0.928 / 29.5%). Say "dev".
- **Ceiling-normalised fragility** divides by each category's *median*
  repeat-call F1.
- **McNemar tests are two-sided exact** (H1/H2 are directional, so the
  reported p-values are conservative; one-sided would halve them).
- **The 12-word overlap check excludes shared instruction templates.** 19
  exam summarisation prompts share the v2 template sentence "Read the
  following article then write a concise summary not exceeding fifty
  words:"; this is intended. No passage is shared.
- **Cost effects vary between prompt sets:** LLMLingua-2 @ 0.5 costs +10.9%
  (v2 dev), +19.0% (v2 test, n = 200) and +12.1% (exam). APCS 1.0.0 costs
  +34.2% (v2 test) and +19.7% (exam). Report the direction as robust and
  the magnitude with its range.
- Unit tests must be run from inside `apcs/` (from the project root,
  Python resolves `apcs` to the outer folder).
