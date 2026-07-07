# Experiment 06 — Findings: APCS calibration (pilot)

**Run:** 2026-07-07 on the 500-prompt pilot (in-sample; dev-only data).

## Finding 1 — The selection problem is real (label distribution, τ = 0.70)

Best-balance labels: llmlingua2@0.7 30.6%, noop 29.0%, llmlingua2@0.5 26.4%,
llmlingua2@0.3 14.0%. No single choice is right for even a third of prompts —
the FrugalGPT-style premise behind APCS (adaptive beats fixed) holds for
Arabic compression. (τ sensitivity: at 0.65 compression is optimal for 88% of
prompts; at 0.75 noop wins 54% — τ shifts the noop/compress boundary, so the
final protocol must pin τ against the measured repeat-call ceiling.)

## Finding 2 — Calibrated rule (shipped as rules_default.json v0.1.0-pilot)

Grid search over the Exp 05 rule template collapsed to a single threshold:

```
token_count < 120  →  no compression
token_count ≥ 120  →  LLMLingua-2 @ rate 0.5
```

(T2 = 200 with r_mid = r_long = 0.5, i.e. the mid/long distinction added
nothing.) Per-category rules gained only +0.6 points accuracy — global rule
kept for simplicity and lower overfitting risk.

## Finding 3 — APCS beats all baselines (primary RQ3 evidence, pilot)

| Policy | Label accuracy | Mean TCR | Mean out-F1 | Fidelity violations (F1<τ) |
|---|---|---|---|---|
| **APCS (global rule)** | **0.412** | 0.271 | 0.846 | 23.6% |
| APCS (per-category) | 0.418 | 0.269 | 0.841 | 22.8% |
| always noop | 0.290 | 0.000 | 1.000 | 0% |
| always llmlingua2@0.7 | 0.306 | 0.328 | 0.735 | 31.6% |
| always llmlingua2@0.5 | 0.264 | 0.530 | 0.672 | 61.4% |
| always llmlingua2@0.3 | 0.140 | 0.727 | 0.613 | 86.0% |
| random selection | 0.240 | 0.396 | 0.752 | 45.8% |
| oracle (label) | 1.000 | 0.340 | 0.822 | 0% |

McNemar exact tests (discordant pairs): APCS vs always-noop p = 7.3e−08;
vs always-llmlingua2@0.7 p = 6.0e−03; vs always-llmlingua2@0.5 p = 8.8e−10.
APCS recovers 80% of the oracle's TCR (0.271 vs 0.340) at comparable mean
fidelity (0.846 vs 0.822) using a one-line explainable rule.

SDR Pareto-hit rate is reported in `policy_comparison.csv` but is degenerate
(always-noop scores 99% — noop is on the frontier for 495/500 prompts, the 5
exceptions being cases where a ~1% LLMLingua-1 trim produced the identical
LLM response); the dissertation should adopt the best-balance label accuracy
as the primary RQ3 metric, with the SDR metric footnoted.

## Finding 4 — The artefact

`apcs` package (installed editable, 10/10 pytest passing): feature extractor
(same code path as calibration data), JSON-configured rule engine, CLI
(`apcs "<prompt>"` / `--json` / `--rules`), library API
(`APCSSelector().recommend()`), optional `compress()` convenience via
LLMLingua-2. Morphological density retained in the extractor but unused by
rules (Exp 05 negative result), off by default.

## Limitations (to carry into the final evaluation)

- In-sample: rules derived and evaluated on the same 500 prompts. The number
  that goes in the dissertation's headline comes from the ~1,000-prompt
  dataset's untouched test split, with rules re-calibrated on its dev split.
- τ = 0.70 provisional (repeat-call ceiling unmeasured).
- 41% accuracy reflects a hard 4-way choice with near-tied candidates; the
  policy-quality metrics (TCR/F1 trade-off vs oracle) are the more meaningful
  performance view.
