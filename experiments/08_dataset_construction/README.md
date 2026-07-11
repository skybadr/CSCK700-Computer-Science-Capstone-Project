# Experiment 08 — AraPromptBench v2 construction (final dataset)

Builds the final 1,000-prompt dataset from the credible public sources named
in the SDR (Table 2), plus 150 synthetic creative prompts. Output:
`../../AraPromptBench_v2.json` (frozen, with per-prompt provenance and split).

## Parts

1. `build_sourced.py` — 850 corpus prompts. Seed-42 sampling, template
   rotation (5 summarisation / 3 QA phrasings), length-band quotas per
   category (deconfounding: short summarisation and long QA included),
   ≥80% Arabic-character gate, normalised-text dedup within v2 and against
   the pilot. `datasets` 5.0 dropped script loaders, so XL-Sum / TyDi-QA /
   ARCD are read from the Hub's `refs/convert/parquet` branches.
2. `creative_prompts_part{1,2,3}.json` — 150 synthetic creative briefs
   (42 short / 73 medium / 35 long after tokenisation), written by Claude
   Fable 5 (a different model family from the gpt-4o-mini under test, to
   avoid generator/evaluator circularity). Every brief caps the requested
   output at ≤120 words (fixes the pilot's creative truncation problem).
3. `assemble_v2.py` — merge, validate, ids (`v2-<cat>-NNN`), stratified
   80/20 dev/test split (seed 42), QC report, 5% author-inspection sample.

## Final composition (1,000)

| Source | n | licence | category |
|---|---|---|---|
| CIDAR | 250 | CC BY-NC 4.0 | instruction |
| Aya (80 MSA + 20 dialect) | 100 | Apache 2.0 | instruction |
| XL-Sum (arabic) | 150 | CC BY-NC-SA 4.0 | summarisation |
| EASC | 100 | research-use | summarisation |
| TyDi-QA (GoldP arabic) | 159 | Apache 2.0 | qa |
| ARCD | 91 | CC BY-SA 4.0 | qa |
| synthetic-claude | 150 | CC BY 4.0 | creative |

Documented deviations from SDR Table 2: ARCD pool exhausted at 91 after
dedup + answer-preserving truncation (QA topped up from TyDi-QA); Aya's
"mixed" rows are all instruction-like and counted there (categories are
therefore 350/250/250/150, not 250×4); creative is synthetic rather than
author-written (enables a synthetic-vs-sourced robustness comparison).

## Contamination guardrail

All 1,000 prompts are new (pilot overlap = 0, enforced by dedup), so the
stratified random test split (200) is untouched by any pilot-era analysis.
The 500-prompt pilot remains dev-only calibration/replication material.

## Before the benchmark runs

The author must review `results/inspection_sample.md` (5% stratified sample,
per the SDR QA plan) and flag any prompt for replacement.
