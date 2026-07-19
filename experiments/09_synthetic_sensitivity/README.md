# Experiment 09 — Synthetic-data sensitivity (advisor request)

**Question:** Does the amount/presence of synthetic data change the benchmark's
conclusions or the APCS calibration? Two parts, both operating on the dev split
only — the 200-prompt test set is never touched by any analysis here.

## 09a — Synthetic-share sweep at 15% / 30% / 50% (no new data, no API cost)

`sweep_09a.py` (ready; runs after the main benchmark). Design: fixed mixes of
240 prompts (60/category — capped by instruction's 30 measured synthetic
prompts, which is exactly the 50%-share requirement), synthetic share
15/30/50% ⇒ 9/18/30 synthetic per category cell, sourced remainder from v2
dev only, 200 bootstrap repeats per share. Each repeat re-runs the Exp 06
grid search; outputs threshold stability (T1 median/IQR, modal rates),
best-balance label composition, and rule accuracy per share.
Stability across shares = robustness evidence; drift = quantified
synthetic-data bias. The v2 test split is never read.

## 09b — Synthetic-vs-sourced matched comparison (deconfounded)

In AraPromptBench v2, synthetic ≡ creative (perfectly confounded). The **probe
pool** (`probe_pool.json`, 105 prompts) breaks this: synthetic instruction /
qa / summarisation prompts, length-banded, written by the same generator as
the v2 creative set (Claude — different model family from the LLM under test).
Comparison of compression outcomes synthetic-vs-sourced *within* category and
band answers "does the benchmark behave differently on synthetic data".

**Quarantine rules:** probe prompts never enter `AraPromptBench_v2.json`,
never inform APCS calibration, never appear in the test set. They are
measured in the SAME benchmark run as v2 (same model snapshot/protocol) so
origin is not confounded with measurement conditions.

**Construction notes** (`finalise_probe.py`): author length estimates drifted
(Arabic ≈ 4.4 cl100k tokens/word), so bands were recomputed from measured
tokens; 11 oversized summarisation passages sentence-trimmed to ≤640 tokens;
12 oversized qa prompts kept intact as `xlong` (trimming risks unanswerable
questions) — used in length-covariate analyses only. 16 genuinely short
qa/summ items added (`probe_short_addendum.json`). 0 duplicates vs pilot/v2.

## Status

Probe pool ready. Both analyses wait on the main v2 benchmark run (which
measures v2 + probe together, after the LLM-under-test decision).
