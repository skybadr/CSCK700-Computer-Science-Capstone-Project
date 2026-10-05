# Analysis and Design

## Chapter Introduction

This chapter explains how the literature and theory of Chapter 2 were turned into a research design and an artefact design. Section 3.2 restates the project specification and maps each component to a research question. Section 3.3 describes the research methodology: design science, a controlled paired experiment, and the safeguards against contamination and post-hoc tuning. Section 3.4 sets out the design decisions for the benchmark, the compression candidates, the LLM protocol, fidelity measurement, labels and metrics, the APCS architecture and the evaluation plan, each with its justification. Where the final design differs from the SDR, this chapter gives the evidence-based reason; Section 4.2.7 consolidates all deviations.

## Project Specification

The project must deliver (i) a benchmark that can measure compression on Arabic, (ii) an analysis pipeline that relates prompt features to compression outcomes, and (iii) an artefact that recommends a compression strategy for any Arabic prompt. Table 2 links each component to the research question it serves, as the SDR required.

| Component | Serves | Specification |
|---|---|---|
| AraPromptBench v2 (1,000 prompts) and exam set (400) | RQ1, RQ3 | MSA prompts from credible public corpora; four task categories; documented licences; deconfounded length bands; dev / test / exam separation with zero overlap |
| Compression layer | RQ1 | LLMLingua, LLMLingua-2 and random deletion at target keep rates 0.7 / 0.5 / 0.3, plus no compression; deterministic and seeded |
| LLM and evaluation module | RQ1–RQ3 | One fixed LLM with deterministic settings; output-level BERTScore with a noise-derived threshold; measured token usage and cost per call |
| Feature extractor | RQ2, RQ3 | Character length, token count, fragmentation ratio, structural complexity, morphological density, task category (plus a length-instruction flag added in Experiment 11) |
| APCS decision engine | RQ3 | Interpretable mapping from features to a strategy; calibrated on dev only; frozen before held-out evaluation |
| Interfaces | Artefact use | Python library API and command-line interface; JSON output; versioned rule files |

Table: Project components and the research questions they serve

The non-functional requirements carried over from the SDR are reproducibility (seeds, configuration and model snapshot recorded with every result), explainability (the selector must state *why* it chose a strategy), low inference overhead (feature extraction without a GPU or an LLM call), and evaluation integrity (held-out data used once, after the design is frozen).

## Research Methods

**Methodology.** The project follows design science research (Hevner et al., 2004). This methodology suits a project that must both build an artefact and generate knowledge about the problem it addresses. Hevner et al.'s guidelines map directly onto the work. *Design as an artefact*: the APCS and AraPromptBench. *Problem relevance*: the Arabic tokeniser tax (Section 2.2). *Design evaluation*: held-out and fresh-exam experiments against baselines. *Research rigour*: paired tests, pre-registration and an independent audit. *Design as a search process*: the iteration from the pilot rule through APCS 1.0.0 to APCS-v2. *Research contributions*: both the artefact and the empirical findings on Arabic compression. *Communication*: this dissertation, the experiment log and a public-ready repository.

**Research approach.** The empirical work is quantitative and experimental. The unit of analysis is a prompt. Every prompt is processed by every strategy, which gives a fully *paired* (within-prompt) design: differences between strategies are not confounded by differences between prompts, and paired tests can be used. Sampling into the benchmark is stratified by source, category and length band with a fixed seed (42), so the dataset is reproducible and its composition controlled.

**Statistical methods.** The choice of tests follows the SDR and standard practice for paired NLP comparisons:

- **Method comparisons (RQ1):** Wilcoxon signed-rank test on per-prompt output F1 differences (Wilcoxon, 1945). This test is non-parametric because F1 differences are bounded and skewed. Paired bootstrap 95% confidence intervals (10,000 resamples) are given for mean differences.
- **Feature effects (RQ2):** Spearman correlations within strata; Kruskal–Wallis tests for category effects within length bands; ordinary least squares regression with standardised features and category indicators. The *unique* variance of each predictor is the drop in R² when it is removed from the full model.
- **Selector accuracy (RQ3):** exact McNemar test on paired correct/incorrect outcomes (McNemar, 1947), with bootstrap 95% confidence intervals for accuracy and accuracy differences.
- **Multiplicity:** Holm's sequentially rejective procedure across the pre-registered hypotheses of Experiment 11 (Holm, 1979).
- **Cost:** one-sided Wilcoxon test on per-prompt cost differences, plus the change in total measured cost.

**Safeguards against contamination and over-fitting.** Evaluation data seen during design inflates measured performance (Sainz et al., 2023; Deng et al., 2024). Four safeguards were applied. (1) The 500-prompt pilot was designated dev-only before any analysis, and AraPromptBench v2 was built with zero overlap with it, so the v2 test split is untouched by pilot-era decisions. (2) The v2 test split was read by a single script, run once, after APCS 1.0.0 was frozen. (3) When Experiment 10's results motivated a redesign, the redesigned selectors were not evaluated on the already-used test split. A new 400-prompt exam set was built instead, with zero exact or 12-word passage overlap with every earlier set. (4) Selectors, hypotheses, comparators and the analysis script for the exam were frozen and committed in a pre-registration document before any exam answer was collected (Appendix D).

**Ethics.** The project involves no human participants, personal data or user prompts. All corpora are public and used under their licences (Table 3), and the creative prompts are synthetic. The proposal records that ethics approval was not required, and nothing in the final design changed that (Appendix C). Two indirect issues are handled explicitly: licences are tracked per prompt, and findings are reported as applying to Modern Standard Arabic and not generalised to all Arabic varieties.

## Design Considerations

### Benchmark design

AraPromptBench v2 follows the composition planned in the SDR (Table 3). Prompts are built from task corpora using fixed templates: five summarisation phrasings and three QA phrasings, rotated to avoid template artefacts. Each prompt is screened by an Arabic-character gate (at least 80% Arabic letters) and deduplicated after Unicode and diacritic normalisation, both within v2 and against the pilot. Length bands are defined in cl100k tokens: short 30–90, medium 91–250, long 251–650.

| Source | Category | Licence | n (v2) | n (exam) |
|---|---|---|---|---|
| CIDAR (Alyafeai et al., 2024) | Instruction | CC BY-NC 4.0 | 250 | 84 |
| Aya, Arabic subset incl. dialectal probe (Singh et al., 2024) | Instruction | Apache 2.0 | 100 | 56 |
| XL-Sum Arabic (Hasan et al., 2021) | Summarisation | CC BY-NC-SA 4.0 | 150 | 60 |
| EASC (El-Haj, Kruschwitz and Fox, 2010) | Summarisation | Research use | 100 | 40 |
| TyDi QA GoldP Arabic (Clark et al., 2020) | QA | Apache 2.0 | 159 | 64 |
| ARCD (Mozannar et al., 2019) | QA | CC BY-SA 4.0 | 91 | 36 |
| Synthetic creative briefs | Creative | CC BY 4.0 | 150 | 60 |
| **Total** | | | **1,000** | **400** |

Table: AraPromptBench v2 and exam set composition

A design weakness found in the pilot drove one deliberate change. In the pilot, almost all short prompts were QA or creative and almost all long ones were summarisation, so *length* and *task type* were confounded and their effects could not be separated. The v2 design therefore fills every category × length cell, deliberately including 57 short summarisation prompts and 99 long QA prompts. This deconfounding is what allows RQ2 to separate the two effects.

The 150 creative prompts are synthetic because public Arabic creative-writing prompts with suitable licences were not available at the required scale. To avoid circularity between generator and evaluator, they were written by a model from a different family from the LLM under test. Every brief caps the requested answer at 120 words or fewer, which fixed the pilot's frequent truncation of creative answers. Because synthetic prompts might behave differently, their effect is measured, not assumed (Experiment 09; Section 5.2.9). The resulting v2 prompts average 220 cl100k tokens (median 185, range 30–664), with a mean fragmentation ratio of 4.1 tokens per word.

The **exam set** (400 prompts: 140 instruction, 100 summarisation, 100 QA, 60 creative) was built with the same sources, templates and band mix, but a different seed (4242). Every v2 record, every v2 QA passage and any text sharing a 12-word span with a pilot, v2 or probe prompt was excluded. Shared instruction templates were exempted because they are identical by design. An independent audit (Section 5.2.10) confirmed zero overlap.

### Compression candidates

Three compressors and a no-compression option were selected from the alternatives in Table 1:

- **LLMLingua-2** (Pan et al., 2024) with its released XLM-RoBERTa-large classifier, representing classifier-based scoring;
- **LLMLingua** (Jiang et al., 2023), representing perplexity-based scoring. Its default GPT-2 scorer was replaced with Qwen2.5-0.5B, because the vertical slice showed that GPT-2 corrupts Arabic (Section 4.2.2);
- **random deletion** at the same keep rate, seeded per prompt: the control that any intelligent method must beat;
- **no compression**, which is always available and always passes the fidelity constraint.

Each compressor runs at target keep rates of 0.7, 0.5 and 0.3, the rates fixed in the SDR. Experiment 11 adds **LLMLingua-2-protected** at the same rates (Section 4.2.6). LongLLMLingua, named in the proposal, was replaced by LLMLingua-2 at the SDR stage. LongLLMLingua's question-aware machinery targets long multi-document retrieval contexts that AraPromptBench does not contain, while LLMLingua-2 gives the scientifically more useful contrast of classifier against perplexity scoring.

### LLM under test and calling protocol

One commercial LLM is used throughout the final benchmark, with the most deterministic settings it accepts: temperature 0 and reasoning effort "none", so no hidden reasoning tokens are billed. The completion limit is 1,024 tokens. The SDR's limit of 512 truncated 9.6% of pilot answers (401 of 4,173), which biased the fidelity scores. The SDR specified gpt-4o-mini, and the pilot used it. By the time of the final benchmark, that model had been retired. The final runs use gpt-5.6-luna, a current small model from the same provider, billed the same way (separate prices per input and output token). A partial run on an intermediate model was stopped by an exhausted API quota; it was archived and is not used, so all reported results come from one model snapshot in one uninterrupted run. Every call records the API's own prompt-token and completion-token counts and the model snapshot. The order of calls is shuffled with a fixed seed so that any drift over time cannot line up with a method or category.

### Fidelity measurement

Three design decisions govern fidelity.

**Output level, not prompt level.** Fidelity is measured between the LLM's answers to the compressed and original prompts, not between the prompts. The pilot showed why: at prompt level, random deletion scores *higher* than LLMLingua-2, because BERTScore rewards keeping words verbatim. At output level the ranking reverses (Section 5.2.2). Prompt-level F1 is retained only as a diagnostic.

**Scorer.** The SDR named multilingual BERT as the primary scorer and AraBERT-v02 as the check. Experiment 01 compared them on 4,000 controlled perturbation pairs. Both rank pairs almost identically (Spearman 0.98), but AraBERT has 50% more dynamic range (0.51 vs 0.34 between identical and unrelated texts). Under multilingual BERT, randomly deleting 30% of a prompt still passes the SDR's 0.85 threshold 86% of the time; under AraBERT, 9.8% of the time. AraBERT therefore became primary (layer 9, no rescaling), consistent with Antoun, Baly and Hajj (2020), and multilingual BERT the sensitivity check.

**Threshold from the noise ceiling.** Each prompt's original version was sent twice, and the two answers scored against each other. τ is the 0.4th percentile of this repeat-call distribution on the dev prompts, rounded to 0.65 for AraBERT (0.75 for multilingual BERT). Section 5.2.1 shows why the SDR's fixed 0.85 could not be kept.

### Labels and metrics

The primary label is **best balance** (Section 2.4): the candidate with the largest TCR whose output F1 ≥ τ, chosen from no compression and LLMLingua-2 at 0.7, 0.5 and 0.3. The SDR's original metric, the share of prompts on which the selector picks *any* Pareto-optimal candidate, was implemented and found to be degenerate. Because no compression always has the highest fidelity, it is always Pareto-optimal, so the trivial policy "never compress" scored 88.5% on the test split, higher than any real selector (Section 5.2.6). Best-balance accuracy keeps the proposal's own definition of "best balance" and removes this degeneracy, so it was adopted as the primary RQ3 metric. Pareto-hit rate is reported as a footnote.

A second label, **cheapest faithful**, is the candidate with the lowest *measured* call cost whose F1 ≥ τ, chosen from ten candidates. It was introduced in Experiment 11 to evaluate the cost-aware selector. Secondary metrics are mean TCR, mean output F1, the rate of answers below τ, the change in total measured cost relative to never compressing, and, for QA prompts, **QA correctness**: whether the answer contains the gold answer string.

### APCS architecture

Figure 1 shows the architecture as built. The APCS is a thin, dependency-light layer: a feature extractor that needs only a tokeniser (plus an optional Java segmenter for morphology), a decision engine that reads its rules from a versioned JSON file, and an optional compression step that applies the recommendation through LLMLingua. The evaluation pipeline is outside the artefact. It produces the evidence from which the rules are calibrated.

![Architecture of the APCS artefact and the evaluation pipeline that calibrates it](figures/fig_architecture.png)

The decision engine was designed in two generations. **APCS 1.0.0** keeps the SDR's form, an ordered list of guarded threshold rules. Calibration searches a grid of token-count thresholds *T1* < *T2* and rates for the "medium" and "long" branches, and picks the rule that maximises best-balance accuracy on dev. The SDR's draft rules also used structural complexity, morphological density and a creative-category clause. They were retained only if the dev evidence supported them; Experiment 05 and Experiment 10 found that it did not (Section 5.2.4). **APCS-v2** keeps interpretability but replaces hand-written guards with a shallow decision tree, learned with the CART algorithm in scikit-learn (Pedregosa et al., 2011) over the full feature set plus task category. Depth and minimum leaf size are chosen by 10-fold stratified cross-validation on dev only. A tree of depth four is still a short list of readable if/then rules (Appendix D), so the SDR's explainability requirement is met. **APCS-cost** uses the same tree learner trained on cheapest-faithful labels, with the protected compressor and LLMLingua among its candidates.

### Evaluation plan and hypotheses

Evaluation proceeds in three stages, each on data the previous stage never saw.

1. **Development (800 prompts):** RQ1 and RQ2 analyses, τ derivation and selector calibration.
2. **Held-out test (200 prompts):** a single evaluation of APCS 1.0.0 against the SDR's baselines: no compression, always LLMLingua-2 (each rate), always LLMLingua (each rate, on labels that admit LLMLingua) and random selection.
3. **Fresh exam (400 prompts):** a pre-registered evaluation of three hypotheses, Holm-corrected:
   - **H1.** APCS-v2 achieves higher best-balance accuracy than always using LLMLingua-2 @ 0.5, the best fixed strategy on dev (exact McNemar).
   - **H2.** APCS-v2 achieves higher accuracy than APCS 1.0.0 (exact McNemar).
   - **H3.** APCS-cost lowers the measured bill relative to never compressing (one-sided Wilcoxon on per-prompt cost differences, and a negative total cost change).

The replication accuracy of APCS 1.0.0 on the exam, the protected compressor's effect and QA correctness were declared as secondary analyses.

## Chapter Summary

This chapter specified the components of the system and linked them to the research questions; described a design-science methodology built on a paired, stratified experiment with explicit safeguards against contamination; and justified each design decision. These covered the deconfounded benchmark, the compression candidates, the LLM protocol, output-level fidelity with an Arabic scorer and a noise-derived threshold, the best-balance and cheapest-faithful labels, the APCS architecture in two generations, and a three-stage evaluation ending in pre-registered hypotheses. Chapter 4 describes how this design was implemented and how it evolved along the way.
