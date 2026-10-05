# Experiment 12 (supplementary) — Compression overhead

**Question:** how long do the APCS recommendation and each compressor take per
prompt? (The proposal motivated compression partly by latency; Experiments
01–11 measured tokens, fidelity and cost but not the compressor's own time.)

**Method:** `measure_overhead.py` — 100 v2 dev prompts (25 per category,
seed 42, mean 210 cl100k tokens), target rate 0.5, five warm-up calls, CUDA
synchronised. APCS `recommend()` (v1 and v2), LLMLingua-2 on GPU and CPU,
LLMLingua-2-protected on GPU, LLMLingua (Qwen2.5-0.5B) on GPU. Workstation:
Intel CPU (24 threads used by torch), RTX 5070. No API calls.

Findings: `results/FINDINGS.md`.
