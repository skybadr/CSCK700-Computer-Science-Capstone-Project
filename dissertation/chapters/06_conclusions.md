# Conclusions

## Chapter Introduction

This final chapter looks back on the project as a whole. Section 6.2 sets out the lessons learned about the topic, about research method and about the project plan. Section 6.3 appraises the project's strengths and weaknesses. Sections 6.4 and 6.5 discuss its academic and business contributions and the limits of each, and extend the findings beyond Arabic. Section 6.6 recommends future research, and Section 6.7 closes the dissertation.

The project set out to discover whether existing prompt compression methods work on Arabic, which prompt features govern their behaviour, and whether a feature-based selector can choose the best strategy per prompt. In brief: classifier-based compression transfers to Arabic with a modest but consistent advantage over chance; task type and length, not morphology, drive the outcome; and a selector that knows a prompt's task category beats every fixed strategy on unseen prompts, though by a modest margin. Along the way the project found that, under current API pricing, compressing a prompt can make it more expensive, and showed how to prevent that.

## Lessons Learned

**About the topic.** The most important lesson is that *cheaper input is not a cheaper call*. The project began, like the literature it built on, by treating token reduction as the goal. The measured bills showed that the compressor's deletions changed the *answer* as well as the prompt. When output is priced well above input, words that carry almost no content, such as "briefly" or "in three sentences", are among the most valuable in the prompt. This lesson generalises beyond Arabic and beyond compression. Any intervention on a prompt, whether compression, translation, templating or truncation, should be evaluated on what it does to the whole exchange, not only to the input.

A second lesson is that **the evaluation instrument must be validated before it is used**. Two choices that looked reasonable in the SDR would have produced wrong conclusions. A fixed threshold of 0.85 would have classified a third of unaltered prompts as damaged. Prompt-level similarity would have ranked random deletion above LLMLingua-2. Both errors were caught only because the pilot measured the instrument itself, through repeat calls and perturbation tests, before measuring the methods. For practitioners, the message is to measure an LLM's own variability before trusting any fidelity threshold, and to compare outputs, not inputs.

A third lesson concerns **intuitions about Arabic**. The project's starting hypothesis, carried into the SDR's decision rules, was that Arabic's rich morphology would be the key to compression behaviour. The data did not support it. What mattered was what the prompt was *for* (its task) and how much text it contained. Once those were known, the tokeniser tax still mattered a little (fragmentation), but morphology did not. Linguistic intuition is a good source of hypotheses, but it must be tested.

**About research method.** Several practices proved their value and would be repeated. Running a cheap pilot on data designated dev-only *before* building the final dataset allowed the design to be corrected without contaminating the held-out evaluation. Recording every experiment with its question, method, results, decision and caveats made the project auditable. It allowed a morphology-measure defect to be found, corrected and reported as an erratum rather than silently absorbed, and it made writing this dissertation largely a matter of assembly. Pre-registering the exam hypotheses removed any temptation to adjust the selectors after seeing the results, and gave the positive H1 and H2 results credibility that an exploratory analysis could not have. The independent audit before writing was cheap insurance: it found three documentation errors, and none in the results.

**About the project plan.** Table 16 compares the SDR's plan with what happened.

| Phase (SDR plan) | Planned | Actual | Comment |
|---|---|---|---|
| Dataset construction | 18 May – 14 Jun | Pilot 5–9 Jul; v2 frozen 11 Jul | About four weeks late |
| Framework implementation | 15 Jun – 12 Jul | Early July, alongside the pilot | Pipeline built incrementally through Exps 01–07 |
| Experimental benchmarking | 13 Jul – 9 Aug | Started July; stopped 20 Jul (API quota); full rerun 29–30 Sep | Two-month interruption; rerun on a newer model |
| APCS design and implementation | 10 Aug – 6 Sep | Pilot package 7 Jul; APCS 1.0.0 30 Sep; redesign 1–5 Oct | Compressed into one week after the rerun |
| Artefact evaluation | 7 – 27 Sep | Test 30 Sep; fresh exam 5 Oct | Exam added beyond the plan |
| Dissertation writing | 21 Sep – 18 Oct | From 5 Oct | Two weeks later than planned |

Table: SDR project plan versus actual progress

The plan's phase order held, but its timing did not. The decisive event was the July interruption. The API account ran out of credit mid-run. Resuming the half-finished run later would have mixed results from two model snapshots, so the benchmark was rerun in full on a single, newer model (gpt-5.6-luna), chosen so that the results would reflect current low-cost models. This was the methodologically correct decision, even though it cost time, and the resumable, deduplicating pipeline made it fast: the entire 13,438-call benchmark ran in about an hour for US$2.49. The SDR's contingency plan ("use smaller models if cost is high") anticipated cost but not quota exhaustion. With hindsight, the plan should have included a hard spending guard, a pre-agreed fallback model, and an earlier full-scale run to surface such risks while there was slack in the schedule.

**Personal growth as a researcher.** This project changed how I approach evidence. I began by trusting the instruments I had specified: the SDR's threshold of 0.85 looked reasonable until a simple repeat-call test showed that a third of answers to *identical* prompts failed it. Since then, my first question about any evaluation has been how much the measurement varies on its own, before asking whether a method works.

I also learned to let data overrule my intuition. The idea at the heart of my SDR was that Arabic's rich morphology would decide how well prompts compress, and the results showed that it did not. Reporting that clearly, rather than searching for an analysis that would rescue it, was uncomfortable at first, but I now see negative results as findings in their own right. The cost result taught a related lesson. I expected compression to save money, and when the bill went up, the satisfying part was not the surprise but tracing it to a mechanism (deleted length instructions) and then fixing it. It made me look at systems as a whole rather than optimising the one number in front of me.

The fresh exam taught me discipline. Writing down my hypotheses and freezing the selectors before seeing a single result removed any temptation to adjust them afterwards, and it is the reason I trust the positive results. Finding a defect in my own morphology measure and filing an erratum taught me that admitting a mistake openly strengthens a piece of work rather than weakening it.

Technically, I gained practical skills in Arabic NLP tools, paired statistical testing and building reproducible pipelines on a budget. Just as importantly, I learned to use AI assistants critically, as collaborators whose output must be checked, which is why the project ends with an independent audit of every number. If I started again, I would add human evaluation of answer quality from the outset and run a full-scale pilot earlier, so that surprises such as the cost effect appear while there is still the most room to act on them.

## Strengths and Weakness of the Project

**Strengths.**

- **Rigour of evaluation.** The paired design, noise-derived threshold, two independent held-out evaluations, pre-registration, Holm correction and an independent audit give the conclusions a strong evidential basis. Negative and partial results (APCS 1.0.0 not beating the best fixed strategy; morphology not predictive) are reported as clearly as the positive ones.
- **Credible, documented data.** 850 of the 1,000 benchmark prompts come from six published, licensed corpora, with per-prompt provenance, deliberate deconfounding of length and task type, and a measured rather than assumed treatment of the synthetic share.
- **Replication.** The ranking flip, method ordering, cost penalty and APCS 1.0.0 accuracy each replicated across independent prompt sets, the last within 0.2 points.
- **A finding of practical value.** The cost penalty, its mechanism and its remedy (the protected compressor) are concrete, testable and immediately useful.
- **An honest, usable artefact.** The APCS is small, explainable, tested, and provably identical to the evaluated selectors.

**Weaknesses.**

- **One LLM.** All final results come from one commercial model. The method ordering is likely to generalise, because it replicated from the pilot model to the final one, but the cost results depend on that model's verbosity and its 6:1 price ratio.
- **Automatic fidelity measure.** BERTScore is a proxy. QA correctness provides an independent check for one category, but no human judgement of answer quality was collected, and creative quality in particular is poorly captured by similarity to a single reference answer.
- **Noisy labels.** Best-balance labels depend on single answers that vary from call to call. With labels this noisy and near-uniform, the attainable accuracy of any selector is limited. Part of the gap between 38.5% and 100% is irreducible noise, not selector error.
- **Modest effect sizes.** APCS-v2's advantage over the best fixed strategy (+5.8 points) is significant but modest, and APCS-cost's saving (1%) is small. A selector is not a large lever on its own.
- **QA.** On QA prompts a fixed 0.5 rate beat both selectors on the exam, so the selectors' features do not yet capture what makes a QA prompt compressible.
- **Coverage.** Dialectal Arabic is represented only by a 2% probe, prompts are at most about 650 tokens long, and the creative category is synthetic.
- **Possible training-data exposure.** The source corpora are public, so the LLM may have seen some passages during training (Golchin and Surdeanu, 2024). This could affect absolute QA correctness, but it should not bias the paired comparisons between compression methods, which all use the same prompts.

## Academic Application and Limitations

The project contributes to research on efficient LLM use in four ways. First, it provides the **first systematic evaluation of prompt compression on Arabic**, establishing that classifier-based compression transfers and perplexity-based compression with English defaults does not. This fills the first gap identified in Section 2.3.6. Second, it shows that **output-level evaluation is necessary**: prompt-level similarity ranks methods backwards. This methodological point applies to compression research in any language, and particularly to morphologically rich ones, where surface-word retention and meaning retention diverge. Third, it provides **evidence on what drives compressibility**: task type and length matter, while morphological density, once controlled, does not. This is consistent with recent findings that morphological type itself confers no intrinsic disadvantage in language modelling (Arnett and Bergen, 2025). Fourth, it identifies the **output-length side-effect of compression**, which connects the compression literature to the literature on length bias and length control (Singhal et al., 2024; Jie et al., 2024) for the first time, and shows how the cost model must change when output is priced above input.

AraPromptBench and its exam set are reusable assets. The documented protocol (noise ceiling, output-level scoring, paired tests, pre-registered fresh evaluation) is a template that can be transferred to other languages.

The academic limitations follow from the scope set in Section 1.3. The conclusions apply to MSA, to prompts of a few hundred tokens, to extractive hard-prompt compression and to one LLM. They do not cover long-context retrieval, soft-prompt methods or dialects. The RQ2 regression explains 15% of variance, so the features studied describe only part of what makes a prompt compressible. The finding that morphology does not matter is conditional on the morphology measure used, average Farasa segments per word, and a richer measure might detect an effect this one missed.

## Business Application and Limitations

For organisations deploying Arabic LLM applications, the findings translate into concrete guidance:

1. **Do not judge compression by tokens saved.** Measure the total bill, input and output, on your own traffic. On output-heavy pricing, naive compression can cost more.
2. **Protect answer-length instructions.** If prompts contain constraints such as "in three sentences" or "no more than fifty words", keep them verbatim. The protected compressor turned a 48% cost increase into an 8% saving on such prompts.
3. **Compress selectively.** Leave short prompts (below roughly 80–130 tokens) alone. Compress long summarisation and instruction prompts. Be cautious with factual QA, where compression measurably reduces correctness.
4. **Choose the selector by objective.** Use APCS-v2 when the aim is the maximum safe token reduction, for example to fit a context window or reduce latency. Use APCS-cost when the aim is the lowest bill at high fidelity.
5. **Use a classifier-based compressor.** LLMLingua-2 was the best method tested. Do not use perplexity-based compression with English scorer models on Arabic.

The commercial value of these measures depends on scale. At the prices studied, the whole 1,000-prompt benchmark cost only a few dollars, so savings matter for high-volume services, not for occasional use. The business limitations mirror the academic ones: the results were measured on one provider's model and price ratio, and the APCS's thresholds should be recalibrated, using the published pipeline, for a different model, a different price ratio or a different prompt mix. Prices and models change quickly; the method for deciding whether compression pays is more durable than any specific number in this dissertation.

**Beyond Arabic.** The core findings are not specific to Arabic and are likely to apply in other settings. Any language that pays a tokeniser tax (Ahia et al., 2023; Petrov et al., 2023) faces the same incentive to compress and the same evaluation pitfalls. The output-length effect depends on pricing, not on language, so it applies to English deployments too. Routing systems that choose between models (Ding et al., 2024; Ong et al., 2025) should account for answer length in the same way. The general lesson is that, in systems where several cost components interact, optimising one component (input tokens) in isolation can worsen the total.

## Recommendations / Prospects for Future Research / Work

1. **Cross-model replication.** Repeat the benchmark on several LLMs with different price ratios and verbosity, including open Arabic models such as ALLaM (Bari et al., 2025), to separate effects of the compressor from effects of the model. The pipeline makes this a matter of changing a configuration file.
2. **Length-aware compressors.** The protected variant is a rule-based patch. A better solution is to train compression classifiers that know which tokens control output length, for example by including answer-length change in the distillation objective of an LLMLingua-2-style model (Pan et al., 2024). An Arabic-specific distillation corpus could also improve the classifier's token-importance judgements.
3. **Better labels and human evaluation.** Use several answers per prompt to reduce label noise, which would raise the ceiling on selector accuracy. Add human judgement of answer quality, at least for creative and summarisation tasks where BERTScore is weakest, or a length-debiased LLM judge validated against a human sample (Zheng et al., 2023; Dubois et al., 2024).
4. **Richer selector features.** Features describing what a QA prompt asks, such as answer type, question position or passage–question overlap, may fix the selectors' weakness on QA. Learned text embeddings could be tested against the interpretable features, trading some explainability for accuracy.
5. **Joint optimisation.** Treat compression strategy, model choice and answer-length control as one decision, in the spirit of FrugalGPT's cascades (Chen, Zaharia and Zou, 2024), optimising expected total cost under a fidelity constraint.
6. **Dialects and longer contexts.** Extend AraPromptBench to major dialects and to long-context and retrieval prompts, where compression ratios, and the case for query-aware methods (Jiang et al., 2024; Nagle et al., 2024), are greatest.
7. **Release.** Publish AraPromptBench, subject to each source corpus's licence, together with the APCS package and its example notebook, so that others can extend the benchmark and recalibrate the selector.

## Chapter Summary

The project answered its three research questions. Existing compression methods are partly effective on Arabic, with classifier-based LLMLingua-2 the best. Task type and length govern compression outcomes, and morphology does not. A feature-based selector can beat every fixed strategy on unseen prompts, by a modest margin. The project delivered AraPromptBench, a reproducible evaluation protocol, and the APCS artefact in three validated variants. Its most practically important finding is that compression can increase the cost of an LLM call by changing the answer, and that protecting answer-length instructions prevents this. The lessons learned (validate the instrument, compare outputs rather than inputs, measure the whole bill, pre-register before testing) apply well beyond Arabic prompt compression. They are offered as the project's contribution to the wider effort to make large language models affordable and fair for speakers of every language.
