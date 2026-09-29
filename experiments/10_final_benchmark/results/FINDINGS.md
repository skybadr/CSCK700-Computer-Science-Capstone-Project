# Experiment 10 — Findings: final benchmark (AraPromptBench v2, gpt-5.6-luna)

**Run:** 2026-09-30. LLM under test **gpt-5.6-luna**, `reasoning_effort=none`,
`temperature=0`, `max_completion_tokens=1024` (the model accepted the most
deterministic configuration). 13,438 calls, **0 failed, one model snapshot
throughout (no drift), $2.49 actual**. 79 answers (0.6%) hit the length cap;
one answer came back empty (a random-deletion variant) and scores 0.
Fidelity τ derived from this model's own noise ceiling: **0.65 (AraBERT)**,
0.75 (mBERT). All RQ1/RQ2/cost numbers are on the 800 dev prompts; APCS test
numbers are from a single evaluation of the 200 untouched test prompts. No
design choice was changed after the test evaluation ran.

Figures: `figures/fig1`–`fig9`; tables: `figures/tables.md`; raw:
`analysis_10.json`, `rq2_final.json`, `cost_analysis.json`, `apcs_final.json`.

## F1 — The noise ceiling replicates; the SDR's 0.85 threshold is untenable again

Two calls with the identical prompt give identical text only **29.6%** of the
time, even at temperature 0 with no reasoning. Median repeat-pair F1 0.930;
**31.8% of identical-prompt pairs fall below 0.85** (pilot, different model:
31.2%). A threshold that rejects a third of pure noise cannot define
acceptable compression. τ = 0.65 is the 0.4th percentile of the dev ceiling,
rounded, so failing it almost certainly reflects real damage.

## F2 — RQ1: LLMLingua-2 is the best compressor for Arabic (replicated)

Output-level F1, LLMLingua-2 minus random deletion, same prompts, paired:

| Target rate | Diff | 95% CI | Wilcoxon p |
|---|---|---|---|
| 0.3 | +0.040 | [0.032, 0.047] | 6e−25 |
| 0.5 | +0.038 | [0.029, 0.046] | 6e−20 |
| 0.7 | +0.050 | [0.041, 0.059] | 2e−29 |

Holds under mBERT (+0.017 to +0.033, all p < 1e−14) and after removing the
2.6% of pairs whose answers exceed AraBERT's 510-wordpiece window (diffs
change by ≤ 0.002).

**The ranking flip replicates:** at prompt level random deletion looks
better (−0.014 / −0.026 / −0.028, all p < 1e−14) because BERTScore rewards
verbatim word retention.

**LLMLingua-2 vs LLMLingua (Qwen scorer), compression-matched** (each prompt
LLMLingua actually compressed, paired with the LLMLingua-2 variant closest in
achieved keep): +0.030 [0.014, 0.045] at keep ≈0.52, +0.041 [0.025, 0.056]
at keep ≈0.66; no difference at the lightest setting, where LLMLingua kept
more text (0.79 vs 0.70). LLMLingua still refuses to compress most prompts
(50–83% returned untouched by category); its length gate is intrinsic
(ρ(tokens, keep) = −0.71 to −0.79).

## F3 — Category fragility, ceiling-normalised (replicates pilot Exp 07 A3)

LLMLingua-2 @ 0.5 output F1 as % of each category's own noise ceiling:
creative 83.7%, summarisation 76.7%, instruction 75.4%, **QA 70.6%**. QA
answers are the most deterministic (ceiling median 1.0) and suffer the most
compression-specific damage; creative answers vary anyway, so compression
costs them least.

## F4 — RQ2: length AND task type both matter; morphology does not

The v2 dataset was built with short summarisation and long QA prompts so
length and category could be separated. LLMLingua-2 @ 0.5, dev:

- **Length helps within every category** (Spearman with output F1: creative
  +0.33, summarisation +0.27, instruction +0.23, all p < 0.001; QA +0.13,
  p = 0.07).
- **Category matters within every length band** (Kruskal–Wallis, short
  p = 0.011, medium p = 1e−13, long p = 3e−8): summarisation most tolerant,
  creative least.
- **OLS with both** (R² 0.148): unique variance explained by category
  **6.5%**, by length **3.4%**. Standardised effects: log length +0.027
  (p = 3e−8); fragmentation −0.012 (p = 0.007); structure −0.003 (n.s.);
  **morphological density +0.002 (p = 0.65)**.

This revises the pilot's "length dominates": in the pilot, short prompts were
almost all QA/creative, so part of what looked like a length effect was a
task-type effect. With the confound broken, both matter and task type
slightly more. Morphology (corrected measure) shows no association once other
features are controlled — the SDR's morphology-based rule is not supported.

## F5 — Compression cut input tokens but raised the total bill

Measured cost of every real call, dev set, vs sending the uncompressed prompt
(output priced 6× input for this model):

| Policy | Input tokens saved | Total cost change | Median answer length vs uncompressed |
|---|---|---|---|
| LLMLingua-2 @ 0.7 | 30% | **+4.6%** | ×1.06 |
| LLMLingua-2 @ 0.5 | 49% | **+10.9%** | ×1.19 |
| LLMLingua-2 @ 0.3 | 68% | **+24.3%** | ×1.52 |
| LLMLingua @ 0.3–0.7 | 8–24% | −2.2 to −3.0% | ×1.00 |
| Random deletion @ 0.3 | 66% | −27.0% | ×1.00 (fidelity 0.55) |

**Mechanism (tested):** 347 dev prompts carry an explicit length
instruction ("answer briefly", "in three sentences", "no more than N
words"). LLMLingua-2 deletes it in 58% / 39% / 21% of cases at rates
0.3 / 0.5 / 0.7. When it is deleted, answers are **1.3–2.4× longer**; when
kept, about 1.1× (Mann–Whitney p ≤ 2e−5 at every rate). Random deletion
removes these instructions just as often, but its garbled prompts do not
produce longer answers. LLMLingua-2 scores brevity words as low-information,
yet they are exactly the tokens that control how much the model writes.

Implication: for models whose output tokens cost several times their input
tokens, compression ratio is not a proxy for cost saving. A compressor (or a
selector) that protects length instructions — or scores cost directly — is
needed. This is a new, practically important finding for the dissertation.

## F6 — RQ3: APCS on the held-out test set (pre-registered protocol)

Calibrated on dev (800): `tokens < 80 → no compression; 80–249 →
LLMLingua-2 @ 0.5; ≥ 250 → LLMLingua-2 @ 0.3` (dev accuracy 38.0%; shipped
as `rules_default.json` v1.0.0-final). Test labels are near-uniform
(0.3: 29%, 0.5: 28.5%, none: 22%, 0.7: 20.5%), which makes this a hard
four-way choice.

| Policy | Test accuracy | Mean TCR | Mean out-F1 | Below τ |
|---|---|---|---|---|
| **APCS** | **31.0%** [24.5, 37.5] | 0.43 | 0.750 | 39% |
| always LLMLingua-2 @ 0.3 | 29.0% | 0.72 | 0.595 | 71% |
| always LLMLingua-2 @ 0.5 | 28.5% | 0.51 | 0.676 | 45% |
| random selection | 27.0% | 0.39 | 0.751 | 36% |
| never compress | 22.0% | 0.00 | 1.000 | 0% |
| always LLMLingua-2 @ 0.7 | 20.5% | 0.30 | 0.752 | 23% |
| always LLMLingua (best rate, extended labels) | 6.0% | 0.11 | 0.883 | 10% |

APCS is **significantly better** than never compressing (+9.0 points,
95% CI [1, 17], McNemar p = 0.038), always LLMLingua-2 @ 0.7 (+10.5,
p = 0.048) and always LLMLingua at every rate (p < 1e−8). It is **not
significantly better** than always LLMLingua-2 @ 0.5 (+2.5, p = 0.61),
@ 0.3 (+2.0, p = 0.75) or random selection (+4.0, p = 0.45). The mBERT
re-run gives the same picture (34%; significant vs never compress, @0.3 and
LLMLingua; not vs @0.5, @0.7 or random). Per category 29–34%.

APCS delivers a better fidelity/savings balance than the aggressive fixed
policies (same TCR class as random selection, far fewer violations than
always-0.3) but, measured by the SDR's primary metric, the one-feature rule
is only marginally better than the best fixed strategy. The pilot's 41% was
in-sample; the gap between dev (38%) and test (31%) is the expected optimism
of in-sample calibration.

Cost: because APCS compresses most prompts with LLMLingua-2, it inherits F5 —
on this model it raises the measured bill by 34% vs never compressing. Its
labels optimise token reduction under a fidelity floor, which F5 shows is not
the same as cost reduction when output is priced 6× input.

**Interpretation for the dissertation (RQ3):** partial support. A
feature-based selector beats naive policies and the perplexity-based
compressor, and recovers most of LLMLingua-2's savings with fewer fidelity
violations, but does not reliably beat the best single LLMLingua-2 setting.
F4 and F5 identify why and what to try next: task type carries more signal
than length, and the target should be measured cost, not token count. These
are reported as future work, not tuned against the test set.

## Protocol integrity

- Test rows read only by `apcs_final.py`, run once in the chain. The later
  re-run of `score_and_analyze.py` (to fix the dev-only LLMLingua comparison,
  F2) re-scored the same responses deterministically; the APCS evaluation was
  not re-run and `apcs_final.json` is from the single original evaluation.
- The one empty answer crashed `bert_score` (removed tokenizer method in
  current `transformers`); empty sides now score 0.0 explicitly.
- The archived July gpt-5.4-mini partial run is not used anywhere.
