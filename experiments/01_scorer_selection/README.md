# Experiment 01 — BERTScore scorer-model selection

**Question:** Which scorer model should be used for semantic-fidelity (BERTScore F1)
measurement in the AraPromptBench compression comparison:
`bert-base-multilingual-cased` (mBERT, SDR primary) or
`aubmindlab/bert-base-arabertv02` (AraBERT, SDR sensitivity check)?

**Why it matters:** All downstream decisions (fidelity threshold F1 ≥ 0.85,
Pareto-optimality labels, APCS rule calibration) depend on this metric. A scorer
that compresses all scores into a narrow band or fails to separate related from
unrelated text would corrupt the ground-truth labels.

## Method

No ground-truth fidelity labels exist, so the comparison uses construct validity.
100 prompts (25 per category, stratified, seed 42) from AraPromptBench v1.1.0.
For each prompt, 8 candidate/reference pairs:

| Perturbation | Expectation for a good scorer |
|---|---|
| identical | F1 ≈ ceiling |
| random word deletion, keep ∈ {0.9, 0.7, 0.5, 0.3} | F1 falls monotonically |
| word shuffle | penalised (word order carries meaning) |
| surface normalisation (punctuation/tatweel strip) | F1 stays near ceiling |
| unrelated prompt (different category) | F1 near floor |

Both models score the identical 800 pairs, layer 9, batch 64.

## Criteria

1. **Monotonicity** — mean per-prompt Spearman ρ between keep-ratio and F1.
2. **Separability** — Cohen's d and pairwise AUC between deletion@0.5 (related)
   and unrelated pairs.
3. **Dynamic range** — mean F1(identical) − mean F1(unrelated).
4. **Robustness** — F1 penalty for meaning-preserving surface edits (lower = better).
5. **Threshold usability** — % of *unrelated* pairs scoring ≥ 0.85 (should be ~0;
   if high, the SDR's 0.85 threshold is meaningless for that scorer) and
   % of mild deletions (keep 0.9) scoring ≥ 0.85 (should be high).

## Run

```
C:\Capstone Project\.venv\Scripts\python.exe scorer_comparison.py
```

Outputs `results/pair_scores.csv` and `results/summary.json`.
