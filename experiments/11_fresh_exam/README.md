# Experiment 11 — Redesigned APCS, evaluated on a fresh exam

**Why:** the v2 test split had been used once (Exp 10), so improved designs
motivated by its results could not be evaluated on it without contamination.
This experiment used a new held-out set to evaluate designs fixed internally
before exam answers were collected.

## Combined release v3

The later [AraPromptBench_v3.json](../../AraPromptBench_v3.json) also
distributes these 400 exam records alongside the 1,000 v2 records. Their
`exam` labels, IDs, answers and source-specific licence fields are
unchanged. The historical scripts and results below continue to use
`AraPromptBench_exam.json`. The combined file does not change the exam's
original evaluation role. See [DATA_PROVENANCE.md](../../DATA_PROVENANCE.md)
for source terms.

## Local workflow (historical execution order)

1. `build_exam.py` → `AraPromptBench_exam.json` (400 prompts; v2 sources,
   templates and band mix; strict exclusion of every v2 record, passage and
   12-word overlap). `audit_exam.py` verifies zero overlap.
   `creative_exam.json`: 60 new synthetic creative briefs, generated with
   Claude Opus 5.5 (Anthropic) on 2026-10-04 (v2 creative set and probe pool:
   Claude Fable 5).
2. QA correctness metric: `../10_final_benchmark/qa_metrics.py`,
   `../10_final_benchmark/qa_correctness.py` (dev).
3. `protect.py` (LLMLingua-2 keeping length-instruction sentences),
   `dev_protected.py` (measured on dev), `design_selectors.py`
   (APCS-v2 and APCS-cost, 10-fold CV on dev) → `results/selectors/`.
4. Fix the local protocol before exam answer collection:
   `PREREGISTRATION.md`, `run_exam.py`, `evaluate_exam.py`
   (code-checked on dev only), `results/selectors/apcs_v1.json`.
5. `run_exam.py` (compress, answer, score) → `evaluate_exam.py` (once) →
   `exam_secondary.py` (reported secondary analyses) →
   `make_exam_figures.py`.

`api_util.py` imports model, parameters and pricing from Exp 10's
`run_api.py` so the protocol is identical. Orchestration, compression,
scoring and analysis were local; the LLM generated answers through its
hosted API.

The historical `PREREGISTRATION.md` file is retained unchanged as a copy of
the internally pre-specified protocol. This submission snapshot was
assembled after evaluation and does not itself establish the protocol's
original timing.

Findings: `results/FINDINGS.md`.
