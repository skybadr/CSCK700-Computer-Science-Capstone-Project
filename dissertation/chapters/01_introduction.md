# Introduction

## Chapter Introduction

Large language models (LLMs) are now accessed mostly through commercial application programming interfaces (APIs) that charge per token, separately for the tokens sent in the prompt and the tokens generated in the answer. For organisations building Arabic-language applications, this pricing model carries a hidden surcharge. Tokenisers trained mainly on English text split Arabic words into many more sub-word pieces than English words, so the same content costs more to process (Ahia et al., 2023; Petrov et al., 2023). Prompt compression, which shortens a prompt before it is sent while trying to keep its meaning, is one of the most practical ways to reduce that cost (Li et al., 2025). Almost all compression methods, however, were designed and evaluated on English.

This dissertation investigates whether existing prompt compression methods work for Arabic and whether a lightweight recommender can choose, for each Arabic prompt, the compression strategy that best balances token savings against fidelity. It presents three linked contributions: AraPromptBench, an Arabic benchmark for prompt compression; an empirical analysis of compression behaviour on Arabic; and the Arabic-Aware Prompt Compression Selector (APCS), a Python artefact that recommends a compression strategy. This chapter states the problem, the research questions, the aims and objectives, the approach taken and the intended outcomes.

## Problem Statement

Practitioners who deploy LLMs for Arabic users face two unresolved problems, first identified in the project proposal and Specification and Design Report (SDR) (Appendices A and B).

The first is a **measurement problem**. Leading compression methods such as LLMLingua (Jiang et al., 2023) and LLMLingua-2 (Pan et al., 2024) report large reductions in prompt length with little loss of task performance, but their evidence comes from English benchmarks and English-trained scoring models. Arabic differs from English in ways that plausibly matter for compression: it is morphologically rich, attaches clitics such as conjunctions, prepositions and pronouns to words, and is written in a script that byte-level tokenisers fragment heavily (Habash, 2010). Whether methods that decide which tokens are "unimportant" in English make sensible decisions in Arabic is an open empirical question.

The second is a **selection problem**. Even if compression helps on average, no single method or compression rate is likely to be best for every prompt. Short questions may tolerate no compression at all, while long articles to be summarised may tolerate heavy compression. Work on cost-efficient LLM use shows that adaptive, per-query strategies outperform fixed ones (Chen, Zaharia and Zou, 2024; Ding et al., 2024), but this principle has not been applied to choosing a compression strategy, and the prompt characteristics that should drive such a choice for Arabic are unknown.

The project addresses both problems with the three research questions fixed in the proposal:

- **RQ1.** How effective are existing prompt compression techniques on Arabic prompts?
- **RQ2.** How do prompt features influence compression performance?
- **RQ3.** Can a recommendation system accurately select the optimal compression strategy?

The proposed solution is an IT artefact, the APCS, that extracts features from an Arabic prompt and recommends a compression method and target rate. Its rules are derived from the benchmark built to answer RQ1 and RQ2.

## Project Aims and Objectives

The aim of the project is to investigate the effectiveness of prompt compression techniques on Arabic prompts, and to design, implement and evaluate an Arabic-Aware Prompt Compression Selector that recommends the most suitable compression strategy for a given Arabic prompt while balancing token reduction against semantic fidelity.

The aim is pursued through five objectives, carried over from the proposal:

1. **Review** the literature on prompt compression, efficient LLM use, multilingual tokenisation, morphology and Arabic natural language processing (NLP) to identify the research gap (Chapter 2).
2. **Design** a benchmarking framework and recommendation architecture, including the structure of AraPromptBench, the prompt features, the evaluation criteria and the APCS decision logic (Chapter 3).
3. **Implement** the system: construct AraPromptBench, integrate the compression methods into a reproducible pipeline and build the APCS as a Python package (Chapter 4).
4. **Evaluate** the compression methods and the APCS using token compression ratio, semantic fidelity, recommendation accuracy and measured cost, against baseline strategies on held-out data (Chapter 5).
5. **Recommend** how Arabic prompt compression should be used in practice, and where it should not (Chapter 6).

**In scope** are Modern Standard Arabic (MSA) prompts in four task categories (instruction following, summarisation, question answering and creative writing); extractive, token-deletion compression methods that work with any black-box LLM API; and one commercial LLM under test, used with fixed, deterministic settings. **Out of scope** are dialectal Arabic beyond a small probe; "soft" compression methods that require access to model internals (Mu, Li and Goodman, 2023; Ge et al., 2024); long-context retrieval-augmented generation; and training new compression models.

## Approach

The project follows the design science research paradigm (Hevner et al., 2004): it builds a purposeful artefact, evaluates its utility rigorously, and reports what was learned both about the artefact and about the problem. The work proceeded through eleven documented experiments in four phases.

1. **Pilot (Experiments 01–07).** A 500-prompt pilot set was used to select the fidelity scorer, test which compression methods run correctly on Arabic, measure LLM output variability and build a first APCS. The pilot was dev-only from the start, so nothing in it informs the held-out evaluations.
2. **Benchmark construction (Experiment 08).** AraPromptBench v2 contains 1,000 prompts: 850 drawn from six public Arabic corpora and 150 synthetic creative prompts. It is split 800/200 into development and held-out test sets.
3. **Final benchmark (Experiments 09–10).** Every prompt was compressed by each method at three target rates, sent to the LLM under test (13,438 API calls) and scored at output level against the answer to the uncompressed prompt. APCS 1.0.0 was calibrated on the 800 development prompts and evaluated once on the 200 test prompts.
4. **Redesign on a fresh exam (Experiment 11).** Results from Experiment 10 motivated an improved selector, a cost-aware selector and a length-protecting compressor. Because the test split had already been used, these were evaluated on a newly built 400-prompt exam set, with hypotheses pre-registered before any exam answer was collected.

Fidelity is measured by BERTScore F1 (Zhang et al., 2020) between the LLM's answer to the compressed prompt and its answer to the original prompt. The pass threshold is derived from the LLM's own repeat-call noise. Methods are compared with paired non-parametric tests, and selectors with exact McNemar tests and bootstrap confidence intervals. Chapters 3 and 4 describe the method and its implementation in detail.

## Outcome

The intended outcomes, as set out in the proposal, were:

- **AraPromptBench**, a reproducible Arabic benchmark for prompt compression with documented provenance, licences and splits;
- **empirical evidence** on how LLMLingua, LLMLingua-2 and a random-deletion baseline behave on Arabic prompts across task types, and which prompt features explain the differences;
- **the APCS artefact**, an open, explainable Python package with a command-line interface and library API, evaluated against no-compression, fixed-method and random-selection baselines;
- **practical guidance** for developers deploying Arabic LLM applications.

All four were delivered. The evaluation also produced an outcome that was not anticipated: on a model whose output tokens cost six times its input tokens, compressing prompts made the total bill *higher*, because the compressor deleted instructions that controlled answer length. This finding reshaped the final design and is treated as a contribution in its own right (Chapters 5 and 6).

## Chapter Summary

This chapter introduced the cost problem that motivates Arabic prompt compression, framed it as a measurement problem and a selection problem, and stated three research questions, an aim and five objectives. It summarised the design-science approach and the eleven experiments through which the work was carried out, and listed the intended outcomes. Chapter 2 reviews the literature on which the project builds and identifies the research gap it fills.
