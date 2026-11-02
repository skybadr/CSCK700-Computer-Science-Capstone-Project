# Experiment 01 — BERTScore scorer-model selection

**Question:** Which scorer model should be used for semantic-fidelity (BERTScore F1)
measurement in the AraPromptBench compression comparison:
`bert-base-multilingual-cased` (mBERT, initially primary) or
`aubmindlab/bert-base-arabertv02` (AraBERT, initially the sensitivity check)?

**Historical pilot context:** The initial fidelity threshold was F1 ≥ 0.85;
Experiments 07 and 10 later replaced it with a repeat-call-calibrated output
threshold. The initial Pareto labels and APCS calibration also depended on
the scorer selected here. A scorer
that compresses all scores into a narrow band or fails to separate related from
unrelated text would corrupt the ground-truth labels.

## Method

No ground-truth fidelity labels exist, so the comparison uses construct validity.
An initial screen used 100 prompts (25 per category, stratified, seed 42)
from AraPromptBench v1.1.0. The completed comparison used all 500 pilot
prompts (125 per category). Each prompt yields 8 candidate/reference pairs:

| Perturbation | Expectation for a good scorer |
|---|---|
| identical | F1 ≈ ceiling |
| random word deletion, keep ∈ {0.9, 0.7, 0.5, 0.3} | F1 falls monotonically |
| word shuffle | penalised (word order carries meaning) |
| surface normalisation (punctuation/tatweel strip) | F1 stays near ceiling |
| unrelated prompt (different category) | F1 near floor |

Both models score the same pairs, using layer 9 and batch 64: 800 pairs
for the initial screen and 4,000 pairs for the full comparison. The main
`pair_scores.csv` and `summary.json` contain the full run; the `_n100` files
retain the initial screen.

## Criteria

1. **Monotonicity** — mean per-prompt Spearman ρ between keep-ratio and F1.
2. **Separability** — Cohen's d and pairwise AUC between deletion@0.5 (related)
   and unrelated pairs.
3. **Dynamic range** — mean F1(identical) − mean F1(unrelated).
4. **Robustness** — F1 penalty for meaning-preserving surface edits (lower = better).
5. **Threshold usability** — % of *unrelated* pairs scoring ≥ 0.85 (should be ~0;
   if high, the initial 0.85 threshold is unsuitable for that scorer) and
   % of mild deletions (keep 0.9) scoring ≥ 0.85 (should be high).

## Run

From the snapshot root, with the project virtual environment activated:

```
python experiments/01_scorer_selection/scorer_comparison.py
```

Outputs `results/pair_scores.csv` and `results/summary.json`.
