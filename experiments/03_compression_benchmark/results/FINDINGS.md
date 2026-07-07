# Experiment 03 — Findings: full local benchmark (500 prompts, 5,000 rows)

**Run:** 2026-07-05, seed 42, all 500 prompts × {noop + 3 methods × rates 0.7/0.5/0.3}.
Runtime 127 s on RTX 5070, **0 errors / 5,000 rows**. Data: `benchmark_results.csv`;
environment: `manifest.json`. Prompt-level fidelity only — no API calls; method
*ranking* is deferred to output-level evaluation (Phase 3) per Experiment 02 Finding 4.

## Headline: mean by method × target rate

| Method | rate | achieved keep | TCR | F1-AraBERT | F1-mBERT |
|---|---|---|---|---|---|
| llmlingua2 | 0.3 | 0.274 | 0.726 | 0.579 | 0.741 |
| llmlingua2 | 0.5 | 0.470 | 0.530 | 0.669 | 0.793 |
| llmlingua2 | 0.7 | 0.672 | 0.328 | 0.769 | 0.850 |
| llmlingua_qwen | 0.3 | 0.789 | 0.211 | 0.864 | 0.905 |
| llmlingua_qwen | 0.5 | 0.846 | 0.154 | 0.876 | 0.914 |
| llmlingua_qwen | 0.7 | 0.905 | 0.095 | 0.905 | 0.935 |
| random_deletion | 0.3 | 0.297 | 0.703 | 0.601 | 0.759 |
| random_deletion | 0.5 | 0.499 | 0.501 | 0.705 | 0.820 |
| random_deletion | 0.7 | 0.696 | 0.304 | 0.812 | 0.881 |

## Finding 1 — LLMLingua-2's rate control is excellent on Arabic at scale

Achieved keep within 0.03 of target on every category (0.46–0.49 at target 0.5).
Latency 0.032 s/prompt. Confirms Experiment 02 Finding 3 on the full dataset.

## Finding 2 — LLMLingua-1's length gate quantified

% of rows returned effectively uncompressed (keep > 0.95), by category:
creative **97.6%**, qa **96.0%** (short prompts, ~71–77 cl100k tokens);
instruction 25.3%, summarisation 18.4% (longer prompts compress partially:
keep ≈ 0.56–0.61 at target 0.3). LLMLingua-1 therefore acts as an *implicit*
"don't compress short prompts" selector — relevant precedent for the APCS
rule `token_count < 80 → NoCompression`, but it makes LLMLingua-1 largely
inoperative on two of four categories.

## Finding 3 — Prompt-level F1: random deletion "beats" LLMLingua-2 (statistically confirmed, interpret with care)

Paired Wilcoxon (same prompts, same target rate), F1-AraBERT:
| rate | llmlingua2 | random | diff | p |
|---|---|---|---|---|
| 0.3 | 0.579 | 0.601 | −0.022 | 1.4e−22 |
| 0.5 | 0.669 | 0.705 | −0.036 | 1.0e−44 |
| 0.7 | 0.769 | 0.812 | −0.042 | 5.8e−58 |

Two mechanical contributors: (a) BERTScore-vs-original rewards verbatim word
retention, favouring random deletion's intact-word sampling over LLMLingua-2's
function-word stripping; (b) LLMLingua-2 compresses slightly harder than
target (keep 0.470 vs random's 0.499 at rate 0.5). **This is why output-level
F1 (Phase 3) — not prompt-level F1 — is the fidelity axis for Pareto labels.**
If prompt-level F1 were used to rank methods, the benchmark would conclude
random deletion is the best compressor, which qualitative inspection contradicts.

## Finding 4 — The F1 ≥ 0.85 threshold is unreachable for real compression at prompt level

% of rows with prompt-level F1-AraBERT ≥ 0.85: llmlingua2 0–0.6% at all rates;
random deletion ≤ 11.6%; llmlingua_qwen 61–63% *only because* most of its rows
are barely compressed (uncompressed rows score ≈ 1.0). Consequence: the SDR's
fidelity threshold is only meaningful on **output-level** F1, where responses
to a well-compressed prompt can match responses to the original even when the
prompt texts differ. This re-scopes the threshold's role: prompt-level F1 is a
descriptive/diagnostic metric; acceptability and Pareto optimality are defined
on output-level F1.

## Status of methods going into Phase 3

| Method | Verdict so far |
|---|---|
| llmlingua2 | Fully operational; the primary intelligent-compression candidate |
| llmlingua_qwen | Operational on instruction/summarisation only (length gate) |
| random_deletion | Baseline, exact rate control by construction |
| noop | Control |

Phase 3 (output-level evaluation with gpt-4o-mini, temperature 0.0,
max_tokens 512) consumes `benchmark_results.csv`'s compressed texts and settles
the actual method ranking. Requires an OpenAI API key; cost estimate to be
computed from actual token counts before launch.
