# Experiment 11 — Redesigned APCS, evaluated on a fresh exam

**Why:** the v2 test split had been used once (Exp 10), so improved designs
motivated by its results could not be evaluated on it without contamination.
This experiment builds a new held-out set and tests pre-registered designs.

## Steps (in order, each committed before the next)

1. `build_exam.py` → `AraPromptBench_exam.json` (400 prompts; v2 sources,
   templates and band mix; strict exclusion of every v2 record, passage and
   12-word overlap). `audit_exam.py` verifies zero overlap.
   `creative_exam.json`: 60 new synthetic creative briefs.
2. QA correctness metric: `../10_final_benchmark/qa_metrics.py`,
   `qa_correctness.py` (dev).
3. `protect.py` (LLMLingua-2 keeping length-instruction sentences),
   `dev_protected.py` (measured on dev), `design_selectors.py`
   (APCS-v2 and APCS-cost, 10-fold CV on dev) → `results/selectors/`.
4. Freeze: `PREREGISTRATION.md`, `run_exam.py`, `evaluate_exam.py`
   (code-checked on dev only), `results/selectors/apcs_v1.json`.
5. `run_exam.py` (compress, answer, score) → `evaluate_exam.py` (once) →
   `exam_secondary.py` (reported secondary analyses) →
   `make_exam_figures.py`.

`api_util.py` imports model, parameters and pricing from Exp 10's
`run_api.py` so the protocol is identical.

Findings: `results/FINDINGS.md`.
