# Experiment 02 — Compression vertical slice (Arabic feasibility check)

**Question:** Do the compression methods planned for the benchmark (LLMLingua,
LLMLingua-2, random deletion) actually work on Arabic prompts, and does
LLMLingua's perplexity scorer need an Arabic-capable substitution?

**Why it matters:** LLMLingua's published results use English-trained small LMs
(GPT-2 / LLaMA) to score token importance. Arabic text fragments heavily in an
English tokenizer, so perplexity-based importance scores may be unreliable.
LLMLingua-2's XLM-RoBERTa classifier is multilingual and should transfer better.
This slice retires that risk *before* the full 500-prompt benchmark is built,
and produces quantitative evidence for the dissertation's RQ1 discussion of
method transferability.

## Method

20 prompts (5/category, stratified, seed 42) × target keep-rates {0.7, 0.5, 0.3}:

| Method | Scorer / mechanism | Rationale |
|---|---|---|
| noop | passthrough | control |
| random_deletion | seed-42 word deletion | baseline |
| llmlingua_gpt2 | GPT-2 perplexity (English) | tests the English-default configuration |
| llmlingua_qwen | Qwen2.5-0.5B perplexity (multilingual, Arabic in training data) | Arabic-capable substitution |

Note: BLOOM-560m was the first choice for the multilingual scorer but is
incompatible with the llmlingua package (`BloomConfig` lacks
`max_position_embeddings`, which `PromptCompressor.load_model` requires) —
documented here as an integration finding; Qwen2.5-0.5B substituted.
| llmlingua2 | XLM-RoBERTa-large classifier (microsoft/llmlingua-2-xlm-roberta-large-meetingbank) | multilingual by design |

Recorded per (prompt, method, rate): cl100k_base token counts, achieved
keep-rate vs target, prompt-level BERTScore F1 (AraBERT primary, mBERT
sensitivity — per Experiment 01), latency, errors, and the compressed Arabic
text itself for qualitative inspection.

## Pass criteria

1. No crashes / empty outputs on Arabic input.
2. Achieved keep-rate tracks the target (compression control works).
3. At matched keep-rate, LLMLingua variants beat random deletion on F1
   (i.e., they select tokens more intelligently than chance).
4. Qualitative check: compressed text remains recognisable Arabic retaining
   the prompt's core intent.

## Run

```
C:\Capstone Project\.venv\Scripts\python.exe vertical_slice.py
```

Outputs `results/slice_results.csv` (includes compressed texts).
