# Experiment 09 — Synthetic-data sensitivity (advisor request)

**Question:** Does the amount/presence of synthetic data change the benchmark's
conclusions or the APCS calibration? Two parts, both operating on the dev split
only — the 200-prompt test set is never touched by any analysis here.

## 09a — Synthetic-share sweep at 15% / 30% / 50%, full dev scale

`sweep_09a.py` (dry-run FEASIBLE; full sweep runs after the main benchmark).
Design upgraded from 240-prompt mixes to **800-prompt mixes** mirroring the
dev split (instruction 280 / qa 200 / summarisation 200 / creative 120) after
the probe pool was expanded to 456 prompts (~380 in-band) for this purpose.
The share applies to the three manipulable categories; creative is
structurally 100% synthetic in v2 and is held constant (120 dev prompts) in
every mix, so between-condition differences are attributable to the
manipulated share alone. Length-confound control: **fixed per-band quotas,
identical across all conditions**, derived from dev proportions and clamped
to synthetic-pool feasibility at the 50% share (instruction 144/102/34,
qa 51/91/58, summarisation 39/78/83). 200 bootstrap repeats per share; each
repeat re-runs the Exp 06 grid search; outputs threshold stability
(T1 median/IQR, modal rates), label composition, accuracy, and mean token
count per condition (verifies the length control held). Known limitation:
at the 50% share a few synthetic band cells are used exhaustively
(qa-long 29/29, summ-medium 39/39), so bootstrap variety there comes from
the sourced side — documented, not hidden. The v2 test split is never read.

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
