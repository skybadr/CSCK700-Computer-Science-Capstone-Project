# Experiment 01 — Findings and Decision

**Final run:** 2026-07-05, seed 42, **all 500 prompts** (125/category), 4,000 pairs
per model, RTX 5070 (torch 2.11.0+cu128). A pilot on 100 prompts
(`pair_scores_n100.csv`, `summary_n100.json`) produced near-identical numbers;
the full-dataset run below is the citable one.
Models: `bert-base-multilingual-cased` (mBERT) and `aubmindlab/bert-base-arabertv02`
(AraBERT), both layer 9, raw F1 (no baseline rescaling).

## Headline results (n = 500)

| Criterion | mBERT | AraBERT | Better |
|---|---|---|---|
| Monotonicity (Spearman, keep-ratio vs F1) | 0.996 | 0.997 | tie |
| Separability AUC (deletion@0.5 vs unrelated) | 1.000 | 0.9999 | tie |
| Separability Cohen's d | 6.12 | 5.40 | mBERT (marginal) |
| Dynamic range (identical − unrelated) | 0.343 | 0.512 | **AraBERT** |
| Shuffle sensitivity (identical − shuffled) | 0.205 | 0.326 | **AraBERT** |
| Robustness penalty (surface normalisation) | 0.067 | 0.091 | mBERT (marginal) |
| Unrelated pairs passing F1 ≥ 0.85 | 0% | 0% | tie |
| Throughput (sentences/sec, RTX 5070) | 723 | 1324 | **AraBERT** |

Cross-scorer agreement across all 4,000 pairs: Pearson 0.981, Spearman 0.981 —
the two models *rank* pairs almost identically, so Pareto-frontier labels are
largely insensitive to the choice. The choice matters for the **fixed 0.85
threshold** and for **discriminative headroom**.

## The decisive evidence: threshold placement

% of *random-deletion* pairs passing the SDR's "acceptable compression" bar (F1 ≥ 0.85):

| keep-ratio | mBERT | AraBERT |
|---|---|---|
| 0.9 | 100% | 99.8% |
| 0.7 | **86.0%** | **9.8%** |
| 0.5 | 14.2% | 0% |
| 0.3 | 0% | 0% |

Under mBERT, randomly deleting 30% of a prompt's words is judged "acceptable
compression" 86% of the time. The threshold therefore cannot distinguish an
intelligent compressor (LLMLingua removing low-information tokens at ratio 0.7)
from the random-deletion baseline at the same ratio — both pass. Under AraBERT,
random deletion at 0.7 fails ~90% of the time, leaving room for a smart
compressor to demonstrate superiority on the fidelity axis. Since fidelity-vs-TCR
Pareto optimality is the ground-truth label for APCS rule calibration, this
headroom directly improves label quality.

AraBERT's dynamic-range advantage is consistent across all four task categories
(0.49–0.53 vs mBERT's 0.33–0.35).

## Decision

**Primary scorer: `aubmindlab/bert-base-arabertv02` (layer 9). Sensitivity
check: `bert-base-multilingual-cased`.** This swaps the roles stated in the SDR
(which had mBERT primary), justified empirically by threshold discriminability
and dynamic range, and consistent with Antoun et al. (2020), who show
Arabic-specific models outperform multilingual BERT on Arabic tasks. The high
cross-scorer correlation (ρ ≈ 0.98) means the sensitivity-check design retains
its purpose: conclusions that hold under both scorers are robust to scorer choice.

Caveats to report in the dissertation:
- Random deletion is a proxy for compression damage; construct validity, not
  ground truth. The 0.85 threshold's meaning is scorer-relative; it was retained
  under AraBERT where it sits at the boundary between mild (keep 0.9, ~100% pass)
  and aggressive (keep 0.7, ~10% pass) random deletion.
- Raw F1 without baseline rescaling (bert-score's rescaling baselines do not
  exist for AraBERT; raw scores keep the two models comparable).
- mBERT is marginally more robust to surface normalisation (penalty 0.067 vs
  0.091) — punctuation stripping alone moves AraBERT ~0.09, so normalisation
  conventions must be held constant between original and compressed prompts
  when scoring.
