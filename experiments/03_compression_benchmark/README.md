# Experiment 03 — Full local compression benchmark (500 prompts)

**Question (RQ1, local half):** How do the compression methods perform on
Arabic prompts across task categories, measured by token compression ratio
and prompt-level semantic fidelity?

**Scope:** All 500 AraPromptBench v1.1.0 prompts, used unsplit as the pilot
set (dataset-strategy decision 2026-07-05: these 500 are permanently dev-only;
the later 1,000-prompt AraPromptBench v2 run held out new prompts). No API
calls are made here; Experiment 04 measured output-level fidelity with
gpt-4o-mini using these compressed texts.

## Grid

| Axis | Values |
|---|---|
| Prompts | 500 (125 × instruction / summarisation / qa / creative) |
| Methods | noop, random_deletion, llmlingua_qwen (Qwen2.5-0.5B), llmlingua2 (XLM-R-large) |
| Target rates | 0.7, 0.5, 0.3 (noop once per prompt) |
| Fidelity | BERTScore F1: AraBERT-v02 primary, mBERT sensitivity (per Experiment 01) |
| Tokens | tiktoken cl100k_base; achieved keep-rate & TCR reported (per Exp 02 Finding 5) |

Total rows: 500 × (1 + 3×3) = 5,000.

Method configuration follows Experiment 02 decisions: GPT-2 scorer excluded
(unusable on Arabic), Qwen2.5-0.5B substituted for LLMLingua-1, and prompt-level
F1 treated as descriptive only — method ranking is established by the
output-level evaluation in Experiment 04.

## Run

From the snapshot root, with the project virtual environment activated:

```
python experiments/03_compression_benchmark/run_benchmark.py
```

Outputs `results/benchmark_results.csv` (master table incl. compressed texts,
which Experiment 04 consumes) and `results/manifest.json` (reproducibility manifest:
library versions, seed, dataset MD5, device, runtime).
