# AraPromptBench — Experiment Log

Chronological index of all experiments. Each folder contains: `README.md`
(question + method), `results/` (raw data + config), `results/FINDINGS.md`
(results, decision, caveats — thesis-ready).

| # | Date | Experiment | Status | Decision / Outcome |
|---|------|-----------|--------|--------------------|
| 01 | 2026-07-05 | [Scorer-model selection](01_scorer_selection/README.md) | ✅ Complete | AraBERT-v02 (layer 9) adopted as primary BERTScore scorer; mBERT demoted to sensitivity check (swap vs SDR, empirically justified). Pilot n=100, final n=500 (4,000 pairs/model). |
| 02 | 2026-07-05 | [Compression vertical slice](02_compression_vertical_slice/README.md) | ✅ Complete | GPT-2 scorer unusable on Arabic (crashes, char corruption) — dropped. LLMLingua-1 runs with Qwen2.5-0.5B but won't compress short prompts (length gate). LLMLingua-2 transfers best (hits all targets, no breakage). Prompt-level F1 alone insufficient to rank methods → output-level eval decisive. |
| 03 | 2026-07-05 | [Full local benchmark](03_compression_benchmark/README.md) | ✅ Complete | 5,000 rows, 0 errors, 127 s. LLMLingua-2 rate control excellent at scale; LLMLingua-1 gate quantified (uncompressed: 97.6% creative / 96% qa rows). Prompt-level F1 ranks random deletion above LLMLingua-2 (Wilcoxon p<1e-22) — mechanical artefact; F1≥0.85 unreachable at prompt level for real compression → threshold + Pareto labels move to output-level F1 (Phase 3). |
| 04 | 2026-07-05 | [Output-level evaluation](04_output_eval/README.md) | ✅ Complete | 4,173 gpt-4o-mini calls, $0.656 actual. **Ranking flips: LLMLingua-2 beats random deletion at every rate (p≤4e−09)** — opposite of prompt-level, vindicating two-level design. LLMLingua-2 also beats LLMLingua-1 like-for-like (0.760 vs 0.662). Category sensitivity: summ > inst > qa > creative. 0.85 threshold strict at output level; repeat-call ceiling flagged for final protocol. |
| 05 | 2026-07-07 | [Feature analysis + Pareto labels](05_feature_analysis/README.md) | ✅ Complete | RQ2: token_count dominates outcomes (ρ +0.4 llmlingua2 fidelity; −0.75 llmlingua-1 gate); **morphological density a weak predictor (ρ≤0.16) — SDR's morphology rule not supported**. Fragmentation ratio 4.2–4.5 tokens/word = tokenizer tax quantified. Frontier: llmlingua2 dominates instruction/summ; light-or-none optimal for short qa/creative. Farasa standalone mode (interactive corrupts Arabic on Windows). Draft APCS rules sketched. |
| 06 | 2026-07-07 | [APCS calibration + artefact](06_apcs_calibration/README.md) | ✅ Complete | RQ3 pilot: best-balance labels split 29/31/26/14% across noop/ll2@0.7/0.5/0.3 → adaptive selection justified. Calibrated rule: `tok<120 → none, else llmlingua2@0.5`. **APCS 41.2% label accuracy beats all fixed baselines (McNemar p≤6e−03)**, recovers 80% of oracle TCR at comparable fidelity. `apcs` package shipped (CLI + API, 10/10 tests). SDR Pareto-hit metric shown degenerate → best-balance accuracy adopted as primary RQ3 metric. In-sample; final numbers await 1,000-prompt held-out split. |
| 07 | 2026-07-09 | [Protocol checks](07_protocol_checks/README.md) | ✅ Complete | (a) Repeat-call ceiling: only 13% identical responses at temp 0; **31% of identical-prompt pairs fail F1≥0.85 → SDR threshold retired for output level; τ=0.70 validated (0.4% noise failure)**. Ceiling-normalised view flips category fragility: QA most damaged, creative least. (b) LLMLingua-1 gate intrinsic — target_token gives identical keeps. (c) AraBERT 512 window: 0.6% of pairs, no score impact — non-issue. |
| 08 | 2026-07-11 | [AraPromptBench v2 construction](08_dataset_construction/README.md) | ✅ Complete | Final 1,000-prompt dataset frozen (`AraPromptBench_v2.json`): 850 sourced from the six SDR corpora (CIDAR 250, XL-Sum 150, EASC 100, TyDi-QA 159, ARCD 91, Aya 100 incl. 20 dialectal) + 150 synthetic creative (Claude-generated, ≤120-word output caps). Length bands deconfounded (57 short summ, 99 long qa). 0 pilot overlap → clean stratified 800/200 dev/test split (seed 42). Awaiting author 5% inspection before benchmark runs. |

## Environment (fixed for all experiments)

- Windows 11, RTX 5070 12 GB, 32 GB RAM
- Python 3.11.9, venv at `C:\Capstone Project\.venv`
- torch 2.11.0+cu128, seed 42 throughout
- Dataset: `AraPromptBench_dataset.json` v1.1.0 (500 MSA prompts, 125/category)
- Dataset strategy: the 500 prompts are pilot/dev-only, used unsplit; the final
  ~1,000-prompt dataset will draw its held-out test split exclusively from new
  prompts (decision 2026-07-05).
