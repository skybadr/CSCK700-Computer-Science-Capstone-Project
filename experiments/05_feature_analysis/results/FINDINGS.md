# Experiment 05 — Findings: prompt features vs compression outcomes (RQ2) + Pareto labels

**Run:** 2026-07-07. Features for all 500 prompts (`features.csv`), Spearman
correlations vs Exp 03/04 outcomes (`correlations.csv`), per-prompt Pareto
frontier over (TCR, output-F1-AraBERT) incl. noop anchor (`pareto_labels.csv`).
Morphological density via Farasa (standalone mode — interactive mode corrupts
Arabic through the Windows pipe; documented integration finding).

> **Erratum (2026-09-29) — morphological density.** The original
> `compute_features.py` divided *all* Farasa segments by the orthographic word
> count. Farasa emits punctuation and numerals as separate tokens, so this
> counted punctuation as extra segments and inflated density (pilot mean 1.85
> vs 1.70 corrected; worked example: 25 segments over 12 Arabic words = 2.08,
> old formula 2.38). The corrected definition counts only tokens containing an
> Arabic letter (`apcs.features.density_from_segmented`, unit-tested).
> Re-check (`recheck_density.py` → `density_recheck.json`): old and corrected
> measures rank prompts similarly (ρ = 0.94), and the corrected correlations
> with output F1 are slightly *stronger* (llmlingua2 ρ = +0.10/+0.15/+0.19 at
> rates 0.3/0.5/0.7, all p < 0.03; random deletion ρ ≈ +0.15–0.18). Finding 2
> therefore stands in its main claim — length dominates, morphology is far
> weaker — but "barely matters" is overstated: morphology has a **small,
> statistically significant** association (|ρ| ≤ 0.19 vs ≈0.4 for
> token_count). Final-dataset RQ2 (Exp 10) uses the corrected measure only.

## Finding 1 — The Arabic "tokenizer tax" quantified on AraPromptBench

Mean fragmentation ratio is **4.2–4.5 cl100k tokens per orthographic word**
across all categories (English text typically runs ≈1.3). This is the
Petrov et al. (2023) unfairness effect measured on this project's own data and
is the economic motivation for Arabic prompt compression in one number.

## Finding 2 — Length dominates; morphology barely matters (surprise)

Spearman ρ of features vs output-level F1 (AraBERT):

| Feature | llmlingua2 (0.3/0.5/0.7) | llmlingua_qwen | random |
|---|---|---|---|
| token_count | **+0.40/+0.38/+0.36** | **−0.74/−0.71/−0.69** | +0.23/+0.19/+0.15 |
| structural_complexity | +0.22/+0.26/+0.33 | −0.32/−0.34/−0.33 | +0.12/+0.17/+0.21 |
| fragmentation_ratio | +0.18/+0.16/+0.16 | −0.30/−0.30/−0.29 | +0.21/+0.11/+0.07 |
| morphological_density | +0.03/+0.10/+0.16 | −0.07/−0.08/−0.08 | +0.14/+0.12/+0.10 |

- For **LLMLingua-2, longer prompts tolerate compression better** (ρ ≈ +0.4):
  redundancy grows with length, so there is more that can be safely removed.
- LLMLingua-1's strong *negative* correlation is the length gate seen from the
  other side: short prompts are returned untouched (F1 ≈ 1), long prompts get
  compressed and drift (feature vs achieved_keep: ρ ≈ −0.75).
- **Morphological density is a weak predictor everywhere** (|ρ| ≤ 0.16). The
  SDR's illustrative rule `morphological_density ≥ HIGH → LLMLingua-2` is NOT
  supported by the data; length, structure, and task category drive outcomes.
  This is an evidence-driven revision to carry into APCS design (RQ3).
- Redundancy: character_length and token_count are rank-identical (ρ = 1.00) —
  APCS needs only token_count. structural_complexity correlates with length
  (0.67) but adds independent signal for llmlingua2 fidelity.

## Finding 3 — Pareto frontier composition supports per-category selection

% of each category's non-noop frontier points, by method:

| Category | llmlingua2 | llmlingua_qwen | random |
|---|---|---|---|
| instruction | **55.5** | 22.6 | 21.9 |
| summarisation | **51.0** | 28.6 | 20.4 |
| qa | 32.8 | **44.7** | 22.5 |
| creative | 31.8 | **44.1** | 24.1 |

llmlingua2@0.3 sits on the frontier for 92% of prompts (maximum-TCR points are
hard to dominate); the *choice-relevant* signal is the composition above. On
long/structured categories (instruction, summarisation) LLMLingua-2 dominates
the frontier. On short categories (qa, creative), LLMLingua-1's frontier share
is mostly its non-compression behaviour in disguise — i.e. **for short prompts,
light-or-no compression is what is actually Pareto-optimal**. Random deletion
is the smallest contributor everywhere (consistent with Exp 04 Finding 1).

Note also that noop is on the frontier for 495/500 prompts (TCR 0, F1 1.0 —
dominated only in 5 summarisation cases where LLMLingua-1 removed ~1% of
tokens and gpt-4o-mini returned the identical response, i.e. strictly-free
compression);
"how much of the frontier is non-noop" varies: creative/qa prompts average
more frontier points (6.5–6.7) than instruction/summarisation (4.1–4.9),
reflecting flatter trade-off curves on short prompts.

## Draft APCS rule sketch (to be calibrated in RQ3)

Data-supported shape, replacing the SDR's illustrative Algorithm 1:
1. `token_count` below a threshold (≈100; covers most qa/creative) →
   NoCompression or LLMLingua-2 @ 0.7 (light) — compression drift outweighs
   savings on short prompts.
2. Long prompts (≳300, summarisation-like) → LLMLingua-2 @ 0.3–0.5
   (aggressive) — best fidelity-per-token-saved; highest absolute savings.
3. Mid-length structured prompts (instruction-like) → LLMLingua-2 @ 0.5.
4. `morphological_density` dropped as a decision feature (Finding 2) unless
   the 1,000-prompt run contradicts; keep in the extractor for the dissertation's
   negative result.
5. LLMLingua-1 never uniquely optimal like-for-like → not recommended by APCS;
   retained in the benchmark as comparison method.

## Caveats

- Correlations are monotonic associations, not causal; category and length are
  confounded by dataset construction (summarisation prompts are long).
- Pareto labels use output-F1 whose ceiling (repeat-call noise floor) is not
  yet measured (flagged in Exp 04) — frontier *composition* is robust to this,
  threshold-based acceptability claims are not.
- Farasa segments = clitic/affix splits; a coarser proxy for morphological
  complexity than full morpheme analysis.
