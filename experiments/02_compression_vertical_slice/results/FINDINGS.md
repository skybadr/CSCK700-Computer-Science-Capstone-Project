# Experiment 02 — Findings: compression methods on Arabic (vertical slice)

**Run:** 2026-07-05, seed 42, 20 prompts (5/category) × rates {0.7, 0.5, 0.3},
RTX 5070. Raw data incl. compressed Arabic texts: `slice_results.csv`.
Supplementary probes run interactively (filter settings; word-breakage counts)
are reported below.

## Headline table (mean over successful runs)

| Method | keep@0.3 | keep@0.5 | keep@0.7 | F1-AraBERT @0.5 | Crashes | Char corruption | Broken-word frac |
|---|---|---|---|---|---|---|---|
| random_deletion | 0.30 | 0.49 | 0.69 | 0.691 | 0 | 0 | 0.000 |
| llmlingua_gpt2 | 0.65 | 0.76 | 0.87 | 0.704 | **15/60** | **38/45** | 0.453 |
| llmlingua_qwen | 0.81 | 0.86 | 0.91 | 0.886 | 0 | 0 | 0.148 |
| llmlingua2 | 0.27 | 0.47 | 0.66 | 0.668 | 0 | 0 | 0.104 |

(keep@r = achieved cl100k keep-rate at target r; broken-word frac = share of
output words not present in the original prompt, a proxy for sub-word breakage.)

## Finding 1 — The English-default LLMLingua configuration is unusable on Arabic

With GPT-2 as perplexity scorer: (a) crashed on **all 15** summarisation runs —
Arabic inflates ~2–3× in GPT-2's byte-level BPE, pushing ~555-cl100k-token
prompts past GPT-2's 1024-token context; (b) produced U+FFFD character
corruption in **38 of 45** surviving outputs (deleting one byte-level fragment
of a multi-byte Arabic character, e.g. زيادة → �يادة); (c) 45% of output words
were fragments not present in the original; (d) rate control failed (kept 65%
when asked for 30%). **Direct RQ1 evidence: published English defaults do not
transfer to Arabic.**

## Finding 2 — LLMLingua-1 has a length gate even with an Arabic-capable scorer

With Qwen2.5-0.5B (multilingual): no crashes, no corruption, but short prompts
(~75 cl100k tokens; QA and creative) are returned **uncompressed** (keep ≈ 1.0)
regardless of target, while longer prompts compress partially (instruction
0.72, summarisation 0.54 at target 0.3). Disabling sentence/context-level
filters does not change this (probe: defaults vs `use_sentence_level_filter=False`
vs `use_context_level_filter=False` vs both — identical keeps of 1.00/1.00/0.30
on two short QA + one instruction prompt). ~15% of output words are still
sub-word fragments. Note: BLOOM-560m was rejected earlier for a package
incompatibility (`BloomConfig` lacks `max_position_embeddings`).

## Finding 3 — LLMLingua-2 transfers best

Hits every target on every category (0.27/0.47/0.66 vs 0.3/0.5/0.7), zero
crashes, zero corruption, lowest word breakage among the LLMLingua family
(10%), latency <0.05 s/prompt on GPU. Its XLM-RoBERTa classifier (multilingual
training incl. Arabic) is the evident reason. Output style is keyword-like:
function words and phrasing removed, content words kept whole and in order.

## Finding 4 — Caveat: prompt-level F1 under-rates intelligent compression

At matched keep-rate, LLMLingua-2 scores *below* random deletion on prompt-level
AraBERT F1 (0.668 vs 0.691 @0.5). This does not mean it is worse: BERTScore
against the original text rewards verbatim retention, and random deletion keeps
a uniform sample of intact words while LLMLingua-2 systematically strips
function words. Which strategy better preserves *task meaning* can only be
settled by output-level evaluation (LLM responses to compressed vs original
prompts) — Phase 3. Qualitative inspection shows LLMLingua-2 retains content
terms but sometimes drops task-critical details (e.g. numbers in a QA table).
**Methodological consequence: prompt-level F1 alone must not be used to rank
methods; output-level F1 is the decisive fidelity metric.**

## Finding 5 — Compression rate is tokenizer-relative

The compressors target a budget in their *own* tokenizer's tokens; TCR is
measured in cl100k_base (billing) tokens. For Arabic these diverge sharply
(GPT-2 asked for 30% delivered 65% in cl100k terms). Achieved cl100k keep-rate,
not the requested rate, must be the reported quantity throughout the benchmark.

## Decisions for the full benchmark (Phase 2)

1. **Drop the GPT-2 scorer** (Finding 1 documents why — thesis evidence).
2. **LLMLingua-1 runs with Qwen2.5-0.5B** as the perplexity-based method; its
   short-prompt refusal is documented behaviour, and per-length analysis will
   reflect it (it independently supports the APCS intuition "very short prompts
   → NoCompression").
3. **LLMLingua-2 confirmed** (microsoft/llmlingua-2-xlm-roberta-large-meetingbank).
4. Baselines unchanged: noop + seed-42 random deletion.
5. Report achieved cl100k keep-rate alongside target rate for every run.
