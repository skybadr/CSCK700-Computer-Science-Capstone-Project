# APPENDICES

## Approved Dissertation Proposal

[[INSERT_DOCX:Abouabdou_Badr_Proposal.docx]]

## Specification & Design Report

[[INSERT_DOCX:Abouabdou_Badr_SpecDesignReport.docx]]

## Ethical Approval Form

No ethical approval application was required. The approved proposal (Appendix A) records "Ethics Approval Needed? No", and the SDR (Appendix B, "Ethical Implications") confirms the determination. The project involved no human participants, surveys, interviews, personal data or user-generated prompts. All prompts were taken from publicly released research corpora under their stated licences, or were generated synthetically, and LLM API calls contained only these prompts. Nothing in the final design changed this determination.

## Detailed Design

### Feature definitions

| Feature | Definition | Used by |
|---|---|---|
| character_length | Characters after Unicode NFC normalisation | Analysis |
| token_count | Tokens under tiktoken cl100k_base | APCS 1.0.0, APCS-v2, APCS-cost |
| word_count | Whitespace-separated words | Analysis |
| fragmentation_ratio | token_count / word_count | APCS-v2 |
| structural_complexity | Count of line breaks, colons, quoted blocks, brackets, bullets and enumerations | APCS-v2 |
| morphological_density | Farasa segments per Arabic word (tokens containing an Arabic letter only) | Analysis (not predictive) |
| has_length_instruction | Prompt contains an Arabic length marker (Section D.5) | APCS-cost |
| task_category | instruction / summarisation / qa / creative (one-hot) | APCS-v2, APCS-cost |

Table: APCS feature definitions

### APCS 1.0.0 rules (rules_default.json)

```
version: 1.0.0-final; calibrated on AraPromptBench v2 dev (800 prompts)
LLM under test: gpt-5.6-luna; fidelity threshold tau = 0.65 (AraBERT)
if token_count < 80   -> no compression
if token_count >= 250 -> LLMLingua-2 @ 0.3
otherwise             -> LLMLingua-2 @ 0.5
```

### APCS-v2 decision tree (depth 4, minimum leaf 20; dev 10-fold CV accuracy 43.5%)

```
cat_creative <= 0.5 (not creative)
|  token_count <= 104.5
|  |  fragmentation_ratio <= 4.85
|  |  |  structural_complexity <= 6.5  -> LLMLingua-2 @ 0.5
|  |  |  structural_complexity >  6.5  -> LLMLingua-2 @ 0.7
|  |  fragmentation_ratio >  4.85      -> no compression
|  token_count > 104.5
|  |  not QA:  token_count <= 243      -> LLMLingua-2 @ 0.5
|  |           token_count >  243      -> LLMLingua-2 @ 0.3
|  |  QA:      token_count <= 433      -> LLMLingua-2 @ 0.5
|  |           token_count >  433      -> LLMLingua-2 @ 0.7
cat_creative > 0.5 (creative)
|  token_count <= 85.5                 -> no compression
|  token_count > 85.5
|  |  fragmentation_ratio <= 4.12      -> LLMLingua-2 @ 0.7
|  |  4.12 < fragmentation_ratio <= 4.43 -> LLMLingua-2 @ 0.5
|  |  fragmentation_ratio >  4.43      -> LLMLingua-2 @ 0.7
```

### APCS-cost decision tree (depth 3, minimum leaf 40; dev 10-fold CV cost change −1.9%)

```
token_count <= 127.5                          -> no compression
token_count > 127.5
|  no length instruction:  not QA             -> no compression
|                          QA                 -> LLMLingua-2 @ 0.5
|  length instruction:     not creative       -> LLMLingua-2-protected @ 0.3
|                          creative           -> LLMLingua-2 @ 0.7
```

(The learned tree contains two further splits, on token count and on morphological density, whose branches all lead to no compression; they are collapsed here.)

### Length-instruction markers

The protected compressor and the has_length_instruction feature use one regular expression over the following Arabic markers: بإيجاز ("briefly"), موجز ("concise"), جملتين ("two sentences"), ثلاث جمل ("three sentences"), جملة واحدة ("one sentence"), فقرة واحدة ("one paragraph"), لا يتجاوز ("not exceeding") and كلمة ("word", as in "fifty words").

### Pre-registration (Experiment 11, committed before the exam run)

Data: 400 never-seen exam prompts with zero exact, 12-word or record overlap with the pilot, v2 or probe sets. Protocol identical to Experiment 10, with τ = 0.65 fixed from Experiment 10. Frozen selectors: APCS 1.0.0 (replication), APCS-v2 (primary) and APCS-cost (extension). Hypotheses, with Holm correction across H1–H3 at α = 0.05:

- **H1 (primary):** APCS-v2 best-balance accuracy > always LLMLingua-2 @ 0.5, the best fixed strategy on dev. Exact McNemar; supported if the Holm-adjusted p < 0.05 and APCS-v2 is ahead.
- **H2:** APCS-v2 accuracy > APCS 1.0.0. Exact McNemar; same criterion.
- **H3 (extension):** APCS-cost lowers the measured total bill relative to never compressing. One-sided Wilcoxon signed-rank test on per-prompt cost differences; supported if the Holm-adjusted p < 0.05 and the total cost change < 0.

Reported regardless of outcome: APCS 1.0.0 replication accuracy; every policy's best-balance and cheapest-faithful accuracy, mean TCR, mean output F1, share below τ, total cost change and QA correctness; APCS-v2 against every other policy (secondary, uncorrected); and accuracy by category. No selector, threshold, label definition or test could be altered after the exam answers were collected. The full document is `experiments/11_fresh_exam/PREREGISTRATION.md` (commit 318d593).

### Worked example: how compression lengthens the answer

Exam prompt exam-qa-043 (76 cl100k tokens) asks, based on a passage, "Where is the headquarters of Forbes magazine?" and begins "answer briefly" (أجب بإيجاز):

بناءً على الفقرة التالية، أجب بإيجاز: تصنيف: شركات مقرها في جيرسي سيتي. السؤال: اين يقع مقر مجلة فوربس؟

Plain LLMLingua-2 @ 0.5 deleted "briefly" and garbled the question:

الفقرة أجب:شركات جيرسي سيتي مقر مجلة فوربس؟

The protected variant kept the instruction sentence verbatim and compressed the rest:

بناءً على الفقرة التالية، أجب بإيجاز: جيرسي سيتي مقر مجلة فوربس؟

| Variant | Billed input tokens | Answer tokens | Call cost (US$ × 10^−6^) | Output F1 |
|---|---|---|---|---|
| Uncompressed | 44 | 23 | 36.4 | 1.000 |
| LLMLingua-2 @ 0.5 | 24 | 85 | 106.8 | 0.638 |
| LLMLingua-2-protected @ 0.5 | 30 | 17 | 26.4 | 0.768 |

Table: Worked example of plain vs protected compression (exam-qa-043)

Plain compression saved 20 input tokens but the model, no longer told to be brief and given a garbled question, wrote a hedged 85-token answer, so the call cost nearly three times as much. The protected variant saved 14 input tokens, kept the answer short and was the cheapest of the three.

## Code used to develop the IT artefact

### Repository and reproduction

All source code, data and results are in the project repository (GitHub: skybadr/CSCK700-Computer-Science-Capstone-Project; access is provided to the examiners, as the module guidance requires). ** Its structure is:

```
README.md, requirements.txt    start here: overview, quick start and pinned dependencies
AraPromptBench_dataset.json     pilot set (500 prompts, dev-only)
AraPromptBench_v2.json          final benchmark (1,000 prompts, 800 dev / 200 test)
AraPromptBench_exam.json        fresh exam set (400 prompts)
apcs/                           the IT artefact (Python package, tests, README)
experiments/
  01_scorer_selection/ ... 12_compression_overhead/   one folder per experiment:
      README.md (question, method), *.py (scripts),
      results/ (raw outputs, configuration, FINDINGS.md)
  EXPERIMENT_LOG.md             chronological index of all experiments
  AUDIT/                        independent verification of every reported number
```

Reproducing the final results: create a Python 3.11 virtual environment and run `pip install -r requirements.txt` (the top-level README gives step-by-step instructions), set the OPENAI_API_KEY environment variable, then run `experiments/10_final_benchmark/run_local.py`, `run_api.py` and `run_chain.py`, followed by the Experiment 11 scripts in the order given in its README. Stored responses make every analysis re-runnable offline. `experiments/AUDIT/verify_results.py` re-checks every headline number from the raw data.

### Experiment log

| # | Experiment | Outcome |
|---|---|---|
| 01 | Scorer selection | AraBERT primary, mBERT sensitivity check |
| 02 | Compression vertical slice | GPT-2 scorer unusable on Arabic; Qwen2.5-0.5B adopted |
| 03 | Local benchmark | Prompt-level F1 ranks random above LLMLingua-2 |
| 04 | Output-level evaluation | Ranking flips at output level |
| 05 | Feature analysis (erratum 05e) | Length dominant in pilot; morphology weak |
| 06 | APCS calibration (pilot) | 41.2% in-sample; Pareto-hit degenerate |
| 07 | Protocol checks | Noise ceiling; 0.85 threshold retired |
| 08 | AraPromptBench v2 | 1,000 prompts frozen, 800/200 split |
| 09 | Synthetic-data sensitivity | Safe for calibration, not for fidelity measurement |
| 10 | Final benchmark | RQ1/RQ2 answered; cost penalty; APCS 1.0.0 31.0% on test |
| 11 | Fresh exam | APCS-v2 38.5%; H1–H3 supported |
| 12 | Compression overhead (supplementary) | Selector ≈0.05 ms; LLMLingua-2 ≈30 ms (GPU), ≈0.4 s (CPU) |

Table: Experiment log summary

### Source code of the APCS package

[[CODE:apcs/apcs/features.py]]

[[CODE:apcs/apcs/selector.py]]

[[CODE:apcs/apcs/protect.py]]
