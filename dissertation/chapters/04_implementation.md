# Implementation (Realisation)

## Chapter Introduction

This chapter describes what was built and how the design of Chapter 3 was realised in practice, including where implementation evidence forced the design to change. Section 4.2 covers the environment, the pilot experiments and the design decisions they triggered, the construction of AraPromptBench v2, the benchmark pipeline, the calibration of APCS 1.0.0, the Experiment 11 redesign, and a consolidated account of every change from the SDR with its justification. Section 4.3 presents the IT artefact itself: the APCS package, its interfaces, its tests and how it is reproduced.

## Details of the Implementation

### Environment and tooling

All work was done in Python 3.11.9 on a Windows 11 workstation with an NVIDIA RTX 5070 GPU (12 GB), which ran the compressors and BERTScore locally. Only LLM calls went to the API. Table 4 lists the main libraries. Every experiment lives in its own folder containing the scripts, a README stating the question and method, a `results/` folder with raw outputs and configuration, and a FINDINGS file with results, decision and caveats. Git and a private GitHub repository recorded each milestone; with one commit per milestone (17 commits on the main branch at the time of writing). A chronological experiment log indexes all eleven experiments (Appendix E).

| Purpose | Library (version) |
|---|---|
| Compression | llmlingua 0.2.2 (LLMLingua, LLMLingua-2) |
| Models and scoring | torch 2.11 (CUDA 12.8), transformers 5.13, bert-score 0.3.13 |
| Tokenisation | tiktoken 0.13 (cl100k_base) |
| Morphology | farasapy 0.1.1 (Farasa, Java 21) |
| LLM API | openai 2.44 (asynchronous client) |
| Analysis | pandas 3.0, scipy 1.17, statsmodels 0.15, scikit-learn 1.9 |
| Testing | pytest |

Table: Implementation environment

Two platform problems had to be solved. First, the GPU's Blackwell architecture required PyTorch builds for CUDA 12.8. Second, Farasa's interactive mode silently corrupted Arabic text passing through the Windows process pipe. It was run in standalone mode instead, which needs a Java runtime but returns correct segmentations.

### Pilot phase and the design changes it triggered

Seven pilot experiments on the 500-prompt pilot set tested each design assumption before money was spent on the final benchmark. The pilot set consists of 500 Modern Standard Arabic prompts, 125 per category, generated with an AI model. Because it was used only for development, its synthetic origin does not affect any held-out result. Table 5 summarises what each found and what it changed.

| Exp. | Question | Finding | Design consequence |
|---|---|---|---|
| 01 | Which BERTScore encoder? | AraBERT and mBERT rank pairs alike (ρ = 0.98), but AraBERT has 50% more dynamic range; under mBERT, random 30% deletion passes 0.85 in 86% of cases | AraBERT primary, mBERT sensitivity check (roles swapped from the SDR) |
| 02 | Do the compressors run on Arabic? | LLMLingua with its default GPT-2 scorer crashed on all long prompts and corrupted characters in 38 of 45 outputs; with Qwen2.5-0.5B it ran cleanly but left short prompts uncompressed; LLMLingua-2 hit all target rates | GPT-2 scorer dropped; Qwen2.5-0.5B adopted; LLMLingua's length gate recorded as a property of the method |
| 03 | Local benchmark at scale | 5,000 compressions, 0 errors; at prompt level random deletion beats LLMLingua-2 (p < 1e−22) | Prompt-level F1 judged a mechanical artefact; fidelity moved to output level |
| 04 | Output-level evaluation | 4,173 API calls (US$0.66); ranking flips: LLMLingua-2 beats random deletion at every rate (p ≤ 4e−9); 9.6% of answers truncated at 512 tokens | Two-level design vindicated; completion limit raised to 1,024 |
| 05 | Feature analysis | Length dominates (ρ ≈ +0.4 with LLMLingua-2 fidelity); morphological density weak (absolute ρ ≤ 0.19); length and category confounded | Morphology rule not adopted; v2 dataset deconfounded by length band |
| 06 | First APCS | Rule "tokens < 120 → none, else LLMLingua-2 @ 0.5": 41.2% best-balance accuracy in-sample; Pareto-hit metric degenerate | Best-balance accuracy adopted as the primary metric; package scaffolded |
| 07 | Protocol checks | Only 13% of repeat calls identical; 31% of identical-prompt pairs score below 0.85 | SDR threshold retired; noise-ceiling τ adopted |

Table: Pilot experiments and the design decisions they produced

Experiment 02 deserves comment because it is direct evidence for RQ1. LLMLingua's published default configuration does not transfer to Arabic. GPT-2's byte-level tokeniser inflates Arabic two- to three-fold, which pushes summarisation prompts past GPT-2's 1,024-token context. When the compressor deletes one byte of a multi-byte Arabic character, the result is an invalid character (for example, زيادة becomes �يادة). Replacing the scorer with the multilingual Qwen2.5-0.5B fixed the corruption, but a length gate remains: short prompts come back almost untouched whatever rate is requested, and disabling the method's sentence- and context-level filters does not change this. The vertical slice also rejected BLOOM-560m as a scorer because it was incompatible with the installed packages.

One defect was found later in the pilot code. The morphological-density feature counted Farasa's punctuation and numeral tokens as words. It was corrected so that only tokens containing an Arabic letter are counted, unit-tested, and the Experiment 05 analysis re-run. Rankings were nearly unchanged (ρ = 0.94 between old and new values), but the association with fidelity became slightly stronger and statistically significant (absolute ρ ≤ 0.19), still far weaker than length. An erratum (05e) records the correction and the softened conclusion.

### Building AraPromptBench v2

`build_sourced.py` draws the 850 corpus prompts with seed 42. It rotates templates, enforces per-category length-band quotas, applies the 80% Arabic-character gate and deduplicates against the pilot after normalisation. For QA, long passages are truncated around the answer span so the question stays answerable. Three of the six corpora (XL-Sum, TyDi QA and ARCD) could no longer be loaded through the `datasets` library's script loaders and were read from the Hugging Face Hub's parquet conversions instead. Two compositional deviations from the SDR resulted. The ARCD pool was exhausted at 91 prompts after deduplication and answer-preserving truncation, so QA was topped up from TyDi QA. And Aya's "mixed" rows were all instruction-like, which made the categories 350/250/250/150 rather than 250 each. `assemble_v2.py` merges the parts, assigns identifiers, draws the stratified 80/20 split and writes a quality report and a 5% stratified inspection sample, which the author reviewed before any benchmark call.

The exam set was built by `build_exam.py` with the same machinery and a different seed. Its first version contained five prompts that partially overlapped earlier passages. A 12-word shingle filter was added and the set rebuilt; `audit_exam.py` then confirmed zero overlap. A 20-prompt sample was reviewed by the author before the exam run.

### The benchmark pipeline

The final benchmark runs in three stages, each a separate script that reads the previous stage's output, so any stage can be re-run without repeating the others.

1. **Local compression (`run_local.py`).** Each prompt is compressed by LLMLingua (Qwen scorer), LLMLingua-2 and random deletion at each target rate. Achieved token counts, TCR and prompt-level F1 under both scorers are recorded. Exp 10 produced 14,560 rows covering the 1,000 v2 prompts and a 456-prompt synthetic probe pool (generated with Claude Fable 5) used by Experiment 09.
2. **API calls (`run_api.py`).** Calls are deduplicated by a hash of (prompt identifier, exact text), so identical compressed texts are sent once. They are shuffled with a fixed seed and sent asynchronously with retries. Each response is appended to a JSON-lines file with the model snapshot, finish reason and the API's token counts, which makes the run resumable. A smoke-test mode checks parameters and projects cost before the full run. The script stops cleanly if the account's quota is exhausted. The final run made **13,438 calls with no failures, one model snapshot throughout and a measured cost of US$2.49**. 79 answers (0.6%) reached the 1,024-token limit. One answer, to a random-deletion variant, came back empty and is scored 0.
3. **Scoring and analysis (`score_and_analyze.py`, `rq2_final.py`, `cost_analysis.py`, `apcs_final.py`).** Output-level BERTScore is computed under both scorers. The repeat-call ceiling and τ are derived, and the RQ1, RQ2 and cost analyses are run on dev. The one-shot test evaluation runs last. `run_chain.py` executes the whole chain in order and logs it.

Two implementation details protect validity. AraBERT's input window is 512 word-pieces. 2.6% of dev answer pairs exceed it, so every RQ1 comparison is repeated without them; the differences change by at most 0.002. The cost of each call is computed from the token counts the API returned for that call, priced at US$0.20 and US$1.20 per million input and output tokens: *cost* = prompt tokens × 0.20/10^6^ + completion tokens × 1.20/10^6^. The cost of a policy is the sum over prompts of the cost of the call for the strategy it chose. Cost is never estimated from TCR.

### Calibrating APCS 1.0.0

`apcs_final.py` computes best-balance labels on the 800 dev prompts, searches a grid of 7 × 6 token-count thresholds and 3 × 3 rates, and writes the winning rule to the package. The rule is:

- token count < 80 → no compression;
- 80 ≤ token count < 250 → LLMLingua-2 @ 0.5;
- token count ≥ 250 → LLMLingua-2 @ 0.3.

Its dev accuracy is 38.0%. The script was written to ship a single global rule, as the SDR specified. Per-category rules were computed for reference only: they raised dev accuracy by 5.6 points in-sample, a gain that, with 120–280 dev prompts per category, could easily be over-fitting. The signal they hinted at was followed up properly, on fresh data, in Experiment 11. The script then read the 200 test prompts once, evaluated APCS 1.0.0 and all baselines, and wrote the result. A later bug fix to the LLMLingua comparison in `score_and_analyze.py` re-scored the same stored answers deterministically. The APCS test evaluation was not re-run.

### The Experiment 11 redesign

Experiment 10 left two clear leads (Chapter 5): task category carried more signal than length, and LLMLingua-2 made answers longer by deleting length instructions. Experiment 11 acted on both. All design work used dev data only.

**Length-protected compression (`protect.py`).** A regular expression identifies Arabic length instructions such as بإيجاز ("briefly"), في ثلاث جمل ("in three sentences") and لا يتجاوز ("not exceeding"). It is the same marker set used to test the mechanism in Experiment 10. The prompt is split into sentence-like segments at sentence punctuation, colons and line breaks. Segments containing a marker are kept verbatim, the rest are compressed with LLMLingua-2 at the target rate, and the pieces are reassembled in order. Prompts without a marker are compressed exactly as plain LLMLingua-2 would compress them. On dev, the variant removed most of the cost penalty and raised fidelity, at the price of a lower achieved TCR. A second idea, protecting question marks in QA prompts, was tested and rejected: dropping the question mark did not reduce QA correctness, so the extra complexity was not justified.

**QA correctness (`qa_metrics.py`).** For QA prompts, an answer is counted correct if it contains any gold answer as a whole-word sequence, after Arabic normalisation: diacritics, tatweel and punctuation removed, and alef, ya and taa-marbuta variants unified. This gives a task-accuracy check that does not depend on BERTScore.

**APCS-v2 and APCS-cost (`design_selectors.py`).** Both are CART decision trees trained on the 800 dev prompts. The features are the five numeric APCS features, the length-instruction flag and one-hot task category. Depth (1–5) and minimum leaf size (10, 20, 40) were chosen by 10-fold stratified cross-validation. APCS-v2 uses best-balance labels and the same four candidates as APCS 1.0.0, so the comparison isolates the selector. The chosen tree (depth 4, minimum leaf 20) reached 43.5% cross-validated accuracy (SD 5.2), against 36.0% for the APCS 1.0.0 rule under the same folds. APCS-cost uses cheapest-faithful labels over ten candidates, including the protected variant and LLMLingua. Its tree (depth 3, minimum leaf 40) cut the cross-validated bill by 1.9%. Both trees were saved as JSON with their readable rules (Appendix D), committed and pre-registered together with `evaluate_exam.py`. The evaluation script had first been run on dev data purely as a code check. `run_exam.py` then compressed, answered and scored the exam: 3,828 calls, no failures, US$0.61. `evaluate_exam.py` ran once.

The learned APCS-v2 tree is easy to read. It first separates creative prompts from the rest. Creative prompts are left uncompressed below 86 tokens and otherwise compressed lightly. For other categories it uses token count, with QA treated separately: long QA prompts are compressed *less*, because their answers depend on details that heavy compression removes. A fragmentation-ratio split leaves short, heavily fragmented prompts uncompressed. APCS-cost is simpler still. It leaves every prompt of up to 127 tokens uncompressed. Above that, prompts without a length instruction stay uncompressed unless they are QA, which get LLMLingua-2 @ 0.5. Prompts with a length instruction get the protected variant @ 0.3, except creative prompts, which get light plain compression (@ 0.7). The selector has in effect learned that compression pays only when the prompt is long and its answer length is either anchored by a protected instruction or naturally short, as in QA.

### Changes from the Specification and Design Report

The rubric expects any change from the SDR to be identified and justified. Table 6 lists every material change. Each was driven by evidence gathered during the project and decided before the data it would be evaluated on was examined.

| SDR specification | As implemented | Evidence and justification |
|---|---|---|
| LLM: gpt-4o-mini, max_tokens 512 | gpt-5.6-luna, temperature 0, reasoning "none", 1,024 completion tokens (pilot used gpt-4o-mini) | A newer model from the same provider gives results that are more relevant and accurate for current deployments; 512 truncated 9.6% of pilot answers (Exp 04). Same billing structure; one snapshot throughout |
| BERTScore: mBERT primary, AraBERT check | AraBERT primary, mBERT check | Exp 01: AraBERT 50% more dynamic range; mBERT passes random 30% deletion at 0.85 in 86% of cases |
| Fidelity threshold F1 ≥ 0.85 | τ = 0.65 from the model's repeat-call noise ceiling (0.75 for mBERT) | Exps 07 and 10: about 31% of *identical-prompt* answer pairs fall below 0.85, so it cannot separate compression damage from noise |
| LLMLingua with its reference scorer | LLMLingua with Qwen2.5-0.5B | Exp 02: GPT-2 crashes and corrupts Arabic; BLOOM incompatible |
| Primary metric: Pareto-hit rate | Best-balance accuracy (proposal's own definition); Pareto-hit as footnote | Exp 06 and Table 11: "never compress" always Pareto-optimal, scores 88.5% |
| Rule engine incl. morphology, structure and creative clauses | APCS 1.0.0: token-count rule; APCS-v2: tree over all features + category | Exps 05 and 10: morphology not predictive when other features are controlled; rules kept only where dev evidence supported them |
| Creative prompts author-written | Synthetic briefs generated with Claude Fable 5 (different provider from the LLM under test) | Scale and licensing; effect measured in Exp 09 |
| Categories 250 × 4 | 350 / 250 / 250 / 150 | ARCD pool exhausted; Aya rows instruction-like (Exp 08) |
| Single held-out evaluation | Held-out test (Exp 10) **plus** fresh pre-registered exam (Exp 11) | Redesign motivated by test results could not be evaluated on the same test without contamination |
| Recommendation of (method, rate) only | Adds APCS-cost and the length-protected compressor | Exp 10: compression raised the total bill; the SDR's cost-saving objective required a cost-aware design |

Table: Changes from the SDR and their justification

None of these changes alters the aim or the research questions. Three of them (the scorer swap, the threshold and the primary metric) were corrections of the SDR's evaluation design that the pilot showed to be necessary. The others kept the study current (the newer LLM), responded to data availability, or followed findings (the cost effect) that the original design could not have anticipated.

## The IT Artefact

The APCS is distributed as an installable Python package (`apcs`, version 1.1.0, MIT licence) in the project repository. Its core dependency is only `tiktoken`. LLMLingua (for applying a recommendation) and farasapy (for morphological density) are optional extras, so recommendations are fast and need no GPU. Table 7 shows the structure.

| Module | Responsibility |
|---|---|
| `features.py` | Normalises the prompt (Unicode NFC) and computes the feature vector: character length, cl100k token count, word count, fragmentation ratio, structural complexity, optional morphological density, and the length-instruction flag |
| `selector.py` | Decision engine. `APCSSelector("v1")` applies the APCS 1.0.0 rules from `rules_default.json`; `"v2"` and `"cost"` walk the frozen decision trees in `selectors/*.json`. Returns a `Recommendation` (method, rate, the guards that fired, the features and the rules version) |
| `protect.py` | The length-protected LLMLingua-2 compressor |
| `cli.py` | Command-line interface: `apcs "<prompt>"` or `apcs --file`, with `--selector`, `--category` and `--json` |
| `rules_default.json`, `selectors/` | Versioned, frozen calibration data, including the dev set it was trained on, τ and the readable rules |

Table: Structure of the APCS package

The decision engine treats rules as data: recalibrating for a different LLM or price ratio means producing a new JSON file, not changing code. Each recommendation states the rule that produced it, which meets the explainability requirement. For example, for a 191-token summarisation prompt from the dev set:

```
$ apcs --selector v2 --category summarisation --file prompt.txt
recommendation : llmlingua2 @ rate 0.5
rule fired     : cat_creative <= 0.5 and token_count > 104.5 and cat_qa <= 0.5 and token_count <= 243
features       : tokens=191 words=41 frag=4.6585 struct=5
rules version  : apcs_v2

$ apcs --selector cost --category summarisation --file prompt.txt
recommendation : llmlingua2_protected @ rate 0.3
rule fired     : token_count > 127.5 and has_length_instruction > 0.5 and cat_creative <= 0.5
```

The two selectors disagree in an informative way. The prompt begins "write a summary *in one paragraph*", a length instruction. APCS-v2, which maximises token reduction, picks plain LLMLingua-2. APCS-cost picks the protected variant, which keeps the instruction and therefore the answer's length. In library form, the same call is:

```python
from apcs import APCSSelector
rec = APCSSelector("cost").recommend(prompt, category="summarisation")
compressed = APCSSelector("cost").compress(prompt, category="summarisation")
```

**Testing.** The package has 18 unit tests. They check feature extraction (empty input rejected, Farasa's punctuation excluded from morphological density), the APCS 1.0.0 rule branches, custom rule files, serialisation, the tree selectors' category handling and their treatment of a missing optional feature. One integration test feeds all 400 exam prompts through the package's own feature extraction and confirms that the shipped APCS-v2 and APCS-cost trees choose *exactly* the strategy recorded in the pre-registered exam evaluation. The artefact a user installs is therefore the one that was evaluated.

**Reproducibility and verification.** Every result file is accompanied by its configuration (model snapshot, parameters, prices, seeds). Before writing began, an independent audit script that imports no experiment code recomputed every headline number in this dissertation from the raw data. It covered the dataset sizes and overlaps, the 13,438 and 3,828 stored responses and their costs, the RQ1–RQ3 statistics and the exam hypotheses, and all 117 checks passed (Section 5.2.10).

## Chapter Summary

The design was implemented as a staged, resumable pipeline around three compressors, one LLM and two BERTScore encoders. Seven pilot experiments tested the design's assumptions and changed it where they failed: the scorer, the threshold, the LLMLingua scorer model, the completion limit and the primary metric. AraPromptBench v2 and the exam set were built from six licensed corpora with documented deviations. APCS 1.0.0 was calibrated on dev and evaluated once on test. Its results motivated a length-protected compressor and two decision-tree selectors, which were pre-registered and evaluated on the fresh exam. All changes from the SDR were listed with their evidence. The artefact is a small, explainable, tested package whose shipped selectors reproduce the evaluated ones exactly. Chapter 5 presents the results.
