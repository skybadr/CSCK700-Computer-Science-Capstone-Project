# Experiment 12 — Findings: compression overhead

**Run:** 2026-10-05, 100 dev prompts (mean 210 tokens), rate 0.5.

| Step | Median ms / prompt | p95 |
|---|---|---|
| APCS recommend (v1 rules) | 0.05 | 0.14 |
| APCS recommend (v2 tree) | 0.05 | 0.14 |
| LLMLingua-2, GPU (RTX 5070) | 29.5 | 31.5 |
| LLMLingua-2-protected, GPU | 28.8 | 30.5 |
| LLMLingua (Qwen2.5-0.5B), GPU | 29.0 | 35.3 |
| LLMLingua-2, CPU | 401 | 408 |

- **The selector is free:** feature extraction plus rule/tree evaluation takes
  about 0.05 ms, so APCS adds no meaningful latency.
- **The compressor is not free:** LLMLingua-2 needs a 560M-parameter encoder;
  about 30 ms per prompt on a consumer GPU and about 0.4 s on CPU. GPU time is
  nearly constant across prompt lengths (fixed per-call cost at these sizes).
- **Protection costs nothing extra** (same time as plain LLMLingua-2).
- **Implication:** for prompts of a few hundred tokens, compression should not
  be expected to reduce end-to-end latency on its own: the compressor's time is
  of the same order as, or larger than, the prompt-processing time it removes,
  and on CPU it adds noticeable delay. API latency savings were not measured
  directly. Compression is justified by cost or context-window limits, not by
  speed, at these prompt lengths.
