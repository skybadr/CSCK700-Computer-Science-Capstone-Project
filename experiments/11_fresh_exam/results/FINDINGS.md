# Experiment 11 — Findings: redesigned APCS on a fresh exam

**Run:** 2026-10-05. 400 never-seen prompts (`AraPromptBench_exam.json`;
zero exact, 12-word partial, record or passage overlap with the pilot, v2 or
probe; author 5% review passed). gpt-5.6-luna, same protocol as Exp 10
(3,828 calls, 0 failed, **$0.61**; Exp 11 total incl. dev work $0.75).
τ = 0.65 fixed from Exp 10. Selectors and evaluation were frozen and
pre-registered (`PREREGISTRATION.md`, commit 318d593) before any exam answer
was collected, and `evaluate_exam.py` ran once.

## Pre-registered results (Holm-corrected across H1–H3)

| Hypothesis | Result | Holm p | Verdict |
|---|---|---|---|
| **H1** APCS-v2 > always LLMLingua-2 @ 0.5 (best fixed on dev) | 38.5% vs 32.8%, **+5.8 pts** [0.8, 10.5] | 0.028 | **Supported** |
| **H2** APCS-v2 > APCS 1.0.0 | 38.5% vs 31.2%, **+7.2 pts** [2.5, 12.0] | 0.008 | **Supported** |
| **H3** APCS-cost lowers the bill vs never compressing | total **−1.0%**; per-prompt Wilcoxon one-sided | 0.001 | **Supported** (small effect; see caveat) |

**R1 replication:** APCS 1.0.0 scores **31.2%** [26.8, 35.8] on the fresh
exam vs 31.0% on the v2 test split — its held-out performance replicates
almost exactly on new data.

## All policies (fresh exam, n = 400)

| Policy | Best-balance accuracy [95% CI] | Mean TCR | Mean out-F1 | Below τ | Cost vs none | QA correct |
|---|---|---|---|---|---|---|
| **APCS-v2** | **38.5%** [33.8, 43.3] | 0.48 | 0.709 | 34.2% | +15.9% | 41% |
| always LLMLingua-2 @ 0.5 | 32.8% [28.3, 37.5] | 0.51 | 0.683 | 39.8% | +12.1% | 40% |
| APCS 1.0.0 | 31.2% [26.8, 35.8] | 0.43 | 0.757 | 34.5% | +19.7% | 40% |
| always LLMLingua-2 @ 0.3 | 30.2% | 0.72 | 0.602 | 69.8% | +27.1% | 32% |
| random selection | 24.5% | 0.38 | 0.765 | 30.5% | +7.5% | 42% |
| always LLMLingua-2 @ 0.7 | 23.2% | 0.31 | 0.765 | 16.0% | +8.4% | 46% |
| **APCS-cost** | 19.5%* | 0.22 | **0.880** | **9.2%** | **−1.0%** | 42% |
| always LLMLingua @ 0.3 | — | 0.17 | 0.862 | 17.5% | −4.0% | 46% |
| never compress | 13.8% | 0.00 | 1.000 | 0% | 0% | 57% |

\* APCS-cost optimises a different target; on its own (cheapest-faithful)
labels it scores 35.7%, the highest of any policy. Fixed protected /
LLMLingua strategies cannot be correct under best-balance labels by
construction and are compared only on cost and fidelity.

APCS-v2 is significantly better than every other strategy (secondary,
uncorrected: vs LLMLingua-2 @ 0.3 +8.3 pts, p = 0.027; vs @ 0.7 +15.2,
p < 1e−4; vs random +14.0, p < 1e−4; vs never +24.7, p < 1e−12).

**By category** (APCS-v2 / APCS 1.0.0 / always @0.5): creative 45 / 23 / 20;
summarisation 53 / 46 / 39; instruction 30 / 24 / 29; **QA 32 / 31 / 40**.
The gain comes from treating creative and summarisation prompts differently;
on QA a fixed 0.5 rate does better than either selector.

## The protected compressor replicates on the exam

On the 180 exam prompts containing a length instruction:

| Rate | Cost, plain → protected | Output F1, plain → protected | TCR, plain → protected |
|---|---|---|---|
| 0.7 | +7.2% → +0.8% | 0.762 → 0.803 | 0.30 → 0.22 |
| 0.5 | +17.6% → **−5.3%** | 0.677 → 0.747 | 0.51 → 0.36 |
| 0.3 | +47.6% → **−7.9%** | 0.597 → 0.687 | 0.71 → 0.51 |

Keeping the length instruction verbatim removes the cost penalty found in
Exp 10 and raises fidelity by 0.04–0.09, at the price of compressing less.
Same pattern as on dev (Exp 11b), now confirmed on unseen prompts.

## QA correctness (answer contains the gold answer)

Uncompressed 57%; LLMLingua 46–53%; LLMLingua-2 32–46%; protected 31–47%;
random deletion 21–31%. Compression costs correctness, more so at higher
rates; ordering of methods matches the similarity results.

## Interpretation for RQ3

- With better features (task category alongside length), a feature-based
  selector **does** beat the best fixed strategy on unseen prompts (H1) and
  improves on the original rule-based APCS (H2). This upgrades the Exp 10
  "partial support" to **support, with a modest effect size** (+5.8 points
  over the best fixed strategy).
- **Accuracy and cost pull in different directions.** APCS-v2 optimises the
  proposal's best-balance target and still raises the bill (+15.9%);
  APCS-cost optimises measured cost and saves only 1% while keeping
  fidelity high (9.2% below τ). A simple fixed LLMLingua @ 0.3 saves more
  (−4.0%) but with twice the low-fidelity answers. Under output-heavy
  pricing, the largest lever is not the selector but the compressor: the
  protected variant is what turns compression into savings.

## Caveats

- H3's pre-registered criterion (one-sided Wilcoxon + negative total) is
  met, but the bootstrap CI for the *mean* per-prompt saving spans zero
  ([−4.6e−6, +2.0e−6] USD): the saving is consistent in direction (median)
  but small in total. Report it as a 1% saving, not a strong one.
- One LLM (gpt-5.6-luna); 400 prompts gives ±4.7-point CIs per policy.
- Creative prompts are synthetic (see Exp 09b caveat).
