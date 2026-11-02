# Experiment 13 (supplementary) — Supplementary analyses and review checks

**Why:** project reviews prompted deeper analysis in several places.
Each question requiring additional evidence is answered
here from the stored results (no new API calls, nothing re-tuned, exam set not
re-used for design).

**Method:** `analyses.py` computes A–K (see the docstring); `results/`
holds `supplementary.json` and `cases.json` (qualitative examples).
Findings: `results/FINDINGS.md`.

**Part 2 (pre-submission review):** `review_checks.py` answers the checks raised
by an independent blind review (L–S, O2: billing tokeniser,
H1 by category, H3 total-bill CI, τ sensitivity, RQ2 with a noise covariate,
per-category lookup, length-marker precision). Output: `results/review_checks.json`;
findings in `results/FINDINGS.md` (Part 2).

**Part 3 (second review, 2026-11-01):** the same script computes T–X: the
rounded threshold tail, category fragility as per-prompt loss, lookup
consequences, dev/test passage overlap and the Arabic-letter gate check.
The optional compression-parity test is in `../../apcs/tests/test_parity.py`.
Outputs remain in `results/review_checks.json`; findings are in Part 3 of
`results/FINDINGS.md`.
