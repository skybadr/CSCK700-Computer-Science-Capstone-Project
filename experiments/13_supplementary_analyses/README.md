# Experiment 13 (supplementary) — Analyses requested in the draft feedback

**Why:** the Dissertation Advisor's feedback on the draft asked for deeper
explanation in several places. Each request that needed evidence is answered
here from the stored results (no new API calls, nothing re-tuned, exam set not
re-used for design).

**Method:** `analyses.py` computes A–K (see the docstring); `results/`
holds `supplementary.json` and `cases.json` (qualitative examples).
Findings: `results/FINDINGS.md`.

**Part 2 (pre-submission review):** `review_checks.py` answers the checks raised
by an independent blind review of the dissertation (L–S, O2: billing tokeniser,
H1 by category, H3 total-bill CI, τ sensitivity, RQ2 with a noise covariate,
per-category lookup, length-marker precision). Output: `results/review_checks.json`;
findings in `results/FINDINGS.md` (Part 2).
