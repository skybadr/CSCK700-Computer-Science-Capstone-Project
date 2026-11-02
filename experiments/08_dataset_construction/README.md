# Experiment 08 — AraPromptBench v2 construction (original benchmark)

Builds the original 1,000-prompt benchmark from the six public research corpora
listed below, plus 150 synthetic creative prompts. Output:
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
   Fable 5 (Anthropic, a different provider from the evaluated OpenAI
   models, reducing direct generator/evaluator circularity). Every brief caps the requested
   output at ≤120 words (fixes the pilot's creative truncation problem).
3. `assemble_v2.py` — merge, validate, ids (`v2-<cat>-NNN`), stratified
   80/20 dev/test split (seed 42), QC report, 5% author-inspection sample.

## Final composition (1,000)

| Source | n | licence | category |
|---|---|---|---|
| CIDAR | 250 | Apache 2.0; research-purpose wording | instruction |
| Aya (80 MSA + 20 dialect) | 100 | Apache 2.0 | instruction |
| XL-Sum (arabic) | 150 | CC BY-NC-SA 4.0 | summarisation |
| EASC | 100 | Terms unresolved for the sampled mirror | summarisation |
| TyDi-QA (GoldP arabic) | 159 | Apache 2.0 | qa |
| ARCD | 91 | MIT declared; incomplete data terms | qa |
| synthetic-claude | 150 | CC BY 4.0 | creative |

These are reviewed status labels, not a certification of rights to the
underlying material. See [DATA_PROVENANCE.md](../../DATA_PROVENANCE.md) for
the checked revisions and remaining uncertainties. `data_licence_status.py`
shares the corrected labels across the v2 and exam builders. Assembly
updates the licence fields even when it reads the historical
`results/sourced_prompts.json`; that raw construction record is retained.

Construction adjustments: ARCD pool exhausted at 91 after
dedup + answer-preserving truncation (QA topped up from TyDi-QA); Aya's
"mixed" rows are all instruction-like and counted there (categories are
therefore 350/250/250/150, not 250×4); creative is synthetic rather than
author-written (enables a synthetic-vs-sourced robustness comparison).

## Contamination guardrail

All 1,000 prompts are new (pilot overlap = 0, enforced by dedup), so the
stratified random test split (200) is untouched by any pilot-era analysis.
The 500-prompt pilot remains dev-only calibration/replication material.

## Pre-run quality review (completed)

The 5% stratified inspection sample is retained in
`results/inspection_sample.md`. The experiment log records that the author's
pre-run inspection passed.

## Combined release v3 (2 November 2026)

`assemble_v3.py` combines the frozen v2 and fresh-exam files into
[AraPromptBench_v3.json](../../AraPromptBench_v3.json): 1,400 prompts with
their original 800 dev / 200 test / 400 exam labels. No record is rewritten
or reassigned. The original v2 builder and files remain the inputs for the
reported experiments. The pilot and synthetic sensitivity pool are not
included in v3.

The combined metadata records both component files, hashes and original
metadata, with separate provenance for the two creative-prompt generators.
Every source and licence field is retained unchanged. Source-specific
terms and unresolved permissions remain documented in
[DATA_PROVENANCE.md](../../DATA_PROVENANCE.md).

Run from the repository root:

```
python experiments/08_dataset_construction/assemble_v3.py
python experiments/08_dataset_construction/assemble_v3.py --check
```
