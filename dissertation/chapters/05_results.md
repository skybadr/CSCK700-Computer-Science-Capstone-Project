# Results and Evaluation

## Chapter Introduction

This chapter evaluates the compression methods and the APCS against the project's aims. As a reminder, the aim was to establish how well existing compression methods work on Arabic (RQ1), which prompt features drive their performance (RQ2), and whether a feature-based selector can choose the best strategy per prompt (RQ3). The evaluation strategy was set out in Section 3.4.7. Results are presented in the order that the logic requires. Section 5.2.1 first establishes that the fidelity measurement is valid. Sections 5.2.2–5.2.5 answer RQ1 and RQ2 on the 800 development prompts and report the effect on cost. Sections 5.2.6–5.2.8 evaluate the APCS on held-out data: once on the v2 test split and once on the pre-registered fresh exam. Sections 5.2.9–5.2.10 test robustness, and Section 5.2.11 answers each research question. Unless stated otherwise, fidelity is output-level BERTScore F1 under AraBERT, τ = 0.65, and costs are measured from billed token counts.

## Evaluation of IT Artefact

### Measurement validity: the noise ceiling

Before compression can be judged, the measuring instrument must be shown to separate compression damage from chance. Sending each uncompressed dev prompt twice to the LLM, at temperature 0 with no reasoning, produced identical text only **29.6%** of the time. The median F1 between the two answers was 0.930, and **31.8% of identical-prompt pairs scored below 0.85** (Figure 2). The pilot model gave almost the same figure (31.2%), so the effect is not specific to one model. This replicates the non-determinism reported by Ouyang et al. (2025) and Song et al. (2025), and has a direct consequence. The SDR's threshold of 0.85 would have classified almost a third of *unaltered* prompts as damaged, and any compressor would have "failed" on many prompts through no fault of its own.

![Repeat-call noise ceiling on the dev set: distribution of output F1 between two answers to the identical prompt, with the SDR threshold (0.85) and the adopted τ (0.65)](figures/fig3_noise_ceiling.png)

τ was therefore set at the 0.4th percentile of this distribution (0.640, rounded to 0.65). Only about 0.4% of noise pairs fall below it, so a compressed prompt that scores below τ has almost certainly been damaged. The ceiling also varies by category. QA answers are the most deterministic (median repeat F1 1.0) and creative answers the least. Raw F1 therefore flatters categories with deterministic answers and penalises creative writing; Section 5.2.3 corrects for this.

### RQ1: effectiveness of compression methods on Arabic

Figure 3 shows the trade-off between tokens removed and output fidelity for each method on the 800 dev prompts. **LLMLingua-2 dominates random deletion at every rate**: at the same achieved TCR it gives higher fidelity, and Table 8 shows the paired differences are highly significant.

![Compression trade-off by method on the dev set: mean TCR against mean output F1 at target rates 0.7, 0.5 and 0.3](figures/fig1_method_tradeoff.png)

| Target rate | LLMLingua-2 | Random deletion | Difference [95% CI] | Wilcoxon p |
|---|---|---|---|---|
| 0.7 | 0.766 | 0.716 | **+0.050** [0.041, 0.059] | 2 × 10^−29^ |
| 0.5 | 0.684 | 0.646 | **+0.038** [0.029, 0.046] | 6 × 10^−20^ |
| 0.3 | 0.594 | 0.554 | **+0.040** [0.032, 0.047] | 6 × 10^−25^ |

Table: Output-level fidelity of LLMLingua-2 vs random deletion (dev, n = 800, paired)

The advantage is real but modest, about 0.04–0.05 F1. LLMLingua-2's classifier, trained on English distillation data, carries a useful notion of token importance into Arabic, but it is far from lossless. At rate 0.5, mean fidelity (0.684) is only just above τ, and at 0.3 it falls below τ on average. Arabic prompts tolerate light compression; heavy compression damages most answers.

**The ranking flip.** At *prompt* level the order reverses. Random deletion scores higher than LLMLingua-2 at every rate (−0.014, −0.026 and −0.028; all p < 10^−14^; Figure 4). BERTScore rewards verbatim retention of surface words, and random deletion keeps whole words scattered evenly through the prompt, while LLMLingua-2 deletes function words and morphological fragments that it rightly judges uninformative. An evaluation that compared prompts, as many compression studies effectively do when they report prompt similarity, would have ranked the methods backwards. This finding, first observed in the pilot (Experiments 03 and 04) and replicated on v2, is a methodological contribution: **for Arabic, compression must be evaluated on the LLM's outputs, not on the compressed prompts.**

![The ranking flip: LLMLingua-2 minus random deletion at prompt level (negative) and output level (positive)](figures/fig2_ranking_flip.png)

**LLMLingua versus LLMLingua-2.** LLMLingua with the Qwen scorer behaves differently. It refuses to compress most prompts: 50–83% are returned untouched, depending on category. Its achieved keep is strongly tied to prompt length (Spearman ρ between tokens and keep from −0.71 to −0.79). This is the intrinsic length gate first seen in Experiment 02. At the same *target* rate the two methods are therefore not comparable, because LLMLingua simply compresses less. When each prompt that LLMLingua did compress is paired with the LLMLingua-2 variant closest in *achieved* keep, LLMLingua-2 is better by +0.030 [0.014, 0.045] at keep ≈ 0.52 and +0.041 [0.025, 0.056] at keep ≈ 0.66. The two do not differ at the lightest setting, where LLMLingua kept more text. Classifier-based scoring therefore transfers to Arabic better than perplexity-based scoring, consistent with Pan et al.'s (2024) argument that unidirectional entropy is poorly aligned with compression.

In summary for RQ1: published English defaults do not transfer (GPT-2 scoring corrupts Arabic); with a multilingual scorer, perplexity-based compression barely compresses short Arabic prompts; and classifier-based LLMLingua-2 is the most effective method, beating random deletion by a modest, consistent margin.

### Category fragility and task accuracy

Dividing each category's mean LLMLingua-2 @ 0.5 fidelity by its own median noise ceiling gives the share of attainable fidelity that survives compression: creative 83.7%, summarisation 76.7%, instruction 75.4% and **QA 70.6%**. On raw F1, creative answers looked the most damaged. Once their natural variability is accounted for, the order reverses: QA answers, which depend on specific facts that compression can delete, lose the most. Creative answers vary anyway, so compression costs them relatively little.

QA correctness on the exam confirms this with a measure independent of BERTScore. Uncompressed prompts produced answers containing the gold answer in **57%** of QA cases. Compression lowered this to 46–53% with LLMLingua, 32–46% with LLMLingua-2 (decreasing with rate), 31–47% with the protected variant and 21–31% with random deletion. The methods rank in the same order as on BERTScore, which supports the validity of the similarity measure, and the size of the drop shows that heavy compression is unsafe for factual QA.

### RQ2: which prompt features matter

Figure 5 shows the rank correlation of each feature with output fidelity at rate 0.5. For LLMLingua-2, longer prompts survive compression better (ρ = +0.28): they contain more redundancy for the compressor to remove. Structural complexity correlates positively (+0.19), fragmentation weakly negatively (−0.12), and morphological density not at all (+0.01). For LLMLingua the length correlation is strongly *negative* (−0.67), because its gate leaves short prompts untouched and therefore perfectly faithful.

![Spearman correlation between prompt features and output fidelity at target rate 0.5 (dev)](figures/fig6_rq2_features.png)

Because v2 was deconfounded by design (Section 3.4.1), length and task type can be separated:

- **Length helps within every category.** Spearman correlations with output F1 are creative +0.33, summarisation +0.27 and instruction +0.23 (all p < 0.001), and QA +0.13 (p = 0.07).
- **Category matters within every length band.** Kruskal–Wallis p = 0.011 (short), 10^−13^ (medium) and 3 × 10^−8^ (long). Summarisation prompts are the most tolerant, creative the least.
- **Joint model.** An OLS regression of output F1 on all features and category explains R² = 0.148. Removing category reduces R² by **6.5 points** and removing length by **3.4 points**. Among standardised features, log length (β = +0.027, p = 3 × 10^−8^) and fragmentation (β = −0.012, p = 0.007) are significant; structure (β = −0.003) and **morphological density (β = +0.002, p = 0.65) are not.**

| Predictor | Standardised β | p | Unique R² |
|---|---|---|---|
| Task category (3 indicators) | — | — | **0.065** |
| log token count | +0.027 | 3 × 10^−8^ | **0.034** |
| Fragmentation ratio | −0.012 | 0.007 | — |
| Structural complexity | −0.003 | n.s. | — |
| Morphological density | +0.002 | 0.65 | — |

Table: OLS model of LLMLingua-2 @ 0.5 output fidelity (dev, R² = 0.148)

Two conclusions follow. First, **task type and length both matter, and task type slightly more.** This revises the pilot's conclusion that length dominated. In the pilot, short prompts were almost all QA or creative, so part of the apparent length effect was a task-type effect. Second, **morphological density, the feature on which the SDR built its Arabic-specific rule, does not predict compression outcomes once length and category are controlled.** This agrees with Arnett and Bergen (2025), who find that morphological type itself confers no intrinsic disadvantage. What does matter is the tokeniser: fragmentation has a small but significant negative effect, as Limisiewicz, Balhar and Mareček (2023) and Ahia et al. (2023) would predict. The low overall R² is itself informative: most of the variation in fidelity is prompt-specific and not captured by these surface features, which caps the accuracy any feature-based selector can reach.

### The cost finding: fewer input tokens, a bigger bill

The proposal's fourth metric was cost saving. Table 10 compares the input tokens saved with the change in the *total* measured bill on the dev set.

| Policy | Input tokens saved | Total cost change | Median answer length vs uncompressed |
|---|---|---|---|
| LLMLingua-2 @ 0.7 | 29.7% | **+4.6%** | ×1.06 |
| LLMLingua-2 @ 0.5 | 49.1% | **+10.9%** | ×1.19 |
| LLMLingua-2 @ 0.3 | 67.7% | **+24.3%** | ×1.52 |
| LLMLingua @ 0.7 / 0.5 / 0.3 | 8.4–23.9% | −2.2 to −3.0% | ×1.00 |
| Random deletion @ 0.3 | 66.1% | −27.0% | ×1.00 (fidelity 0.55) |

Table: Input-token saving vs total measured cost change (dev, output priced at 6× input)

LLMLingua-2 removed up to two-thirds of the input tokens and still **raised** the bill by 5–24% (Figure 6). The cause is output length. Answers to LLMLingua-2-compressed prompts were longer, by a median factor of up to 1.5, and every output token costs six times an input token. Following the cost model of Section 2.4, a modest growth in a typically longer answer outweighs a large cut in the prompt.

![Input tokens saved vs total API bill saved per policy (dev; negative bill saving = costs more)](figures/fig9_cost_input_vs_total.png)

**The mechanism was tested, not assumed.** 347 dev prompts contain an explicit length instruction ("answer briefly", "in three sentences", "no more than fifty words"). LLMLingua-2 deleted that instruction in 58%, 39% and 21% of cases at rates 0.3, 0.5 and 0.7. When it was deleted, answers were **1.3–2.4 times longer**; when it was kept, about 1.1 times (Mann–Whitney p ≤ 2 × 10^−5^ at every rate). The classifier scores words such as "briefly" or "three" as low-information, and in a sense they are: they carry little *content*. But they are exactly the tokens that control *how much* the model writes, which is consistent with the length-control literature (Jie et al., 2024; Han et al., 2025). Random deletion removes length instructions just as often, but its garbled prompts produce shorter, lower-quality answers, so it "saves" money only by destroying fidelity.

This is a new finding with practical weight. For any model whose output tokens cost several times its input tokens, which is common in current commercial pricing, **compression ratio is not a proxy for cost saving**, and a compressor can increase the bill it was meant to reduce. The finding motivated the cost-aware selector and the protected compressor evaluated below.

### RQ3 (stage 1): APCS 1.0.0 on the held-out test split

APCS 1.0.0 (Section 4.2.5) was evaluated once on the 200 untouched test prompts. The test labels were close to uniform (LLMLingua-2 @ 0.3: 29%, @ 0.5: 28.5%, none: 22%, @ 0.7: 20.5%), which makes this a hard four-way choice in which the best fixed strategy can reach at most 29%.

| Policy | Accuracy [95% CI] | Mean TCR | Mean out-F1 | Below τ | Cost vs none | Pareto-hit |
|---|---|---|---|---|---|---|
| **APCS 1.0.0** | **31.0%** [24.5, 37.5] | 0.43 | 0.750 | 39.0% | +34.2% | 79.5% |
| always LLMLingua-2 @ 0.3 | 29.0% | 0.72 | 0.595 | 71.0% | +37.8% | 86.5% |
| always LLMLingua-2 @ 0.5 | 28.5% | 0.51 | 0.676 | 45.0% | +19.0% | 66.5% |
| random selection | 27.0% | 0.39 | 0.751 | 35.5% | +14.3% | 75.5% |
| never compress | 22.0% | 0.00 | 1.000 | 0.0% | 0.0% | **88.5%** |
| always LLMLingua-2 @ 0.7 | 20.5% | 0.30 | 0.752 | 22.5% | +9.6% | 64.0% |
| always LLMLingua (best rate)‡ | 6.0% | 0.11 | 0.883 | 9.5% | −1.2% | 78.0% |

Table: Selector and baseline performance on the held-out test split (n = 200)

‡ Compared on extended labels that admit LLMLingua candidates; under the primary labels it can never be correct.

APCS 1.0.0 had the highest accuracy and was **significantly better** than never compressing (+9.0 points, 95% CI [1, 17], McNemar p = 0.038), always LLMLingua-2 @ 0.7 (+10.5, p = 0.048) and always LLMLingua at every rate (p < 10^−8^). It was **not significantly better** than always LLMLingua-2 @ 0.5 (+2.5, p = 0.61), always @ 0.3 (+2.0, p = 0.75) or random selection (+4.0, p = 0.45). The multilingual BERT re-run, with its own τ and recalibrated rule, gave the same picture (34% accuracy). APCS 1.0.0 delivered a sensible balance: the TCR class of random selection, far fewer fidelity violations than always compressing at 0.3, and per-category accuracy between 29% and 34%. But measured by the primary metric, a one-feature rule was only marginally better than the best fixed strategy. The drop from 38.0% on dev to 31.0% on test is the expected optimism of in-sample calibration.

The Pareto-hit column shows why the SDR's metric was replaced. "Never compress" scores 88.5%, the highest of any policy, because its fidelity of 1.0 can never be dominated. A metric that rewards doing nothing cannot evaluate a compression selector.

![Confusion matrix of APCS 1.0.0 recommendations against best-balance labels (test)](figures/fig5_confusion_matrix.png)

The confusion matrix (Figure 7) shows where the rule fails. It never recommends LLMLingua-2 @ 0.7, although that was the best choice for 20.5% of test prompts. It also recommends @ 0.5 for many prompts whose best choice was no compression or @ 0.3. Token count alone cannot tell these apart. RQ2 had already indicated what was missing, namely task category, and the cost analysis indicated that the target itself (token reduction) was misaligned with the bill. Both were followed up in Experiment 11, on fresh data.

### RQ3 (stage 2): the redesigned selectors on the fresh exam

All three selectors were evaluated once on the 400 never-seen exam prompts under the pre-registered protocol. Table 12 gives the results and Figure 8 the accuracies.

| Hypothesis | Result | Holm p | Verdict |
|---|---|---|---|
| H1: APCS-v2 > always LLMLingua-2 @ 0.5 | 38.5% vs 32.8%, **+5.8 points** [0.8, 10.5] | 0.028 | **Supported** |
| H2: APCS-v2 > APCS 1.0.0 | 38.5% vs 31.2%, **+7.2 points** [2.5, 12.0] | 0.008 | **Supported** |
| H3: APCS-cost bill < never compress | total **−1.0%**; per-prompt one-sided Wilcoxon | 0.001 | **Supported** (small effect) |

Table: Pre-registered hypotheses on the fresh exam (n = 400; exact two-sided McNemar for H1 and H2, conservative for directional hypotheses)

![Best-balance accuracy of each policy on the fresh exam, with 95% bootstrap confidence intervals](figures/exam_fig1_selector_accuracy.png)

| Policy | Accuracy [95% CI] | Mean TCR | Mean out-F1 | Below τ | Cost vs none | QA correct |
|---|---|---|---|---|---|---|
| **APCS-v2** | **38.5%** [33.8, 43.3] | 0.48 | 0.709 | 34.2% | +15.9% | 41% |
| always LLMLingua-2 @ 0.5 | 32.8% [28.3, 37.5] | 0.51 | 0.683 | 39.8% | +12.1% | 40% |
| APCS 1.0.0 | 31.2% [26.8, 35.8] | 0.43 | 0.757 | 34.5% | +19.7% | 40% |
| always LLMLingua-2 @ 0.3 | 30.2% | 0.72 | 0.602 | 69.8% | +27.1% | 32% |
| random selection | 24.5% | 0.38 | 0.765 | 30.5% | +7.5% | 42% |
| always LLMLingua-2 @ 0.7 | 23.2% | 0.31 | 0.765 | 16.0% | +8.4% | 46% |
| **APCS-cost** | 19.5%† | 0.22 | **0.880** | **9.2%** | **−1.0%** | 42% |
| always LLMLingua @ 0.3 | — | 0.17 | 0.862 | 17.5% | −4.0% | 46% |
| never compress | 13.8% | 0.00 | 1.000 | 0.0% | 0.0% | 57% |

Table: All policies on the fresh exam (n = 400)

† APCS-cost optimises a different target. On its own cheapest-faithful labels it scores 35.7%, the highest of any policy.

**H1 and H2.** APCS-v2 was the most accurate policy and significantly better than every alternative. Beyond the pre-registered comparisons, it beat always @ 0.3 by +8.3 points (p = 0.027), always @ 0.7 by +15.2 (p < 10^−4^), random selection by +14.0 (p < 10^−4^) and never compressing by +24.7 (p < 10^−12^); these secondary tests are uncorrected. It also had fewer fidelity violations than the best fixed strategy (34.2% vs 39.8%). The by-category breakdown shows where the gain comes from. For APCS-v2, APCS 1.0.0 and always @ 0.5 respectively, accuracy was creative 45 / 23 / 20%, summarisation 53 / 46 / 39%, instruction 30 / 24 / 29% and QA 32 / 31 / 40%. Treating creative and summarisation prompts differently produces the gain. On QA, a fixed 0.5 rate still does better than either selector, a limitation discussed in Chapter 6.

**Replication.** APCS 1.0.0 scored 31.2% on the exam against 31.0% on the test split. Its held-out performance replicated almost exactly on independent data, which supports the reliability of both evaluations.

**H3 and the accuracy–cost tension.** APCS-cost met its pre-registered criterion: the per-prompt cost difference was significantly negative and the total fell by 1.0%. It did so while keeping fidelity high (mean 0.880; only 9.2% of answers below τ, the lowest of any compressing policy). The effect, however, is small. The bootstrap interval for the *mean* per-prompt saving spans zero, so the saving is consistent in direction but modest in total, and is reported as such. APCS-v2, which optimises the proposal's best-balance target, raised the bill by 15.9%. A simple fixed LLMLingua @ 0.3 saved more (−4.0%), but with nearly twice APCS-cost's rate of low-fidelity answers. **Accuracy against the best-balance label and cost pull in different directions**, and no single selector maximises both.

### The length-protected compressor

The protected compressor was evaluated on the 180 exam prompts that contain a length instruction (Table 14; Figure 9).

| Target rate | Cost: plain → protected | Output F1: plain → protected | TCR: plain → protected |
|---|---|---|---|
| 0.7 | +7.2% → +0.8% | 0.762 → 0.803 | 0.30 → 0.22 |
| 0.5 | +17.6% → **−5.3%** | 0.677 → 0.747 | 0.51 → 0.36 |
| 0.3 | +47.6% → **−7.9%** | 0.597 → 0.687 | 0.71 → 0.51 |

Table: Plain vs length-protected LLMLingua-2 on exam prompts with a length instruction (n = 180)

![Total API cost relative to the uncompressed prompt, plain vs protected LLMLingua-2 (exam prompts with a length instruction)](figures/exam_fig2_protected_cost.png)

Keeping the length instruction verbatim turned a cost *increase* of up to 48% into a *saving* of 5–8% at the two heavier rates. It also raised fidelity by 0.04–0.09, at the price of compressing less. The same pattern had been found on dev before the exam was run, so this is a replicated result. It shows that under output-weighted pricing the largest lever is the *compressor*, not the selector: protecting the few tokens that control answer length is what turns compression into a saving.

### Synthetic data sensitivity

Experiment 09 tested whether the 15% synthetic share of v2 could have biased the results, in response to a question from the dissertation advisor. **(a) Calibration:** APCS was recalibrated on 200 bootstrap mixes at 15%, 30% and 50% synthetic share, with category and length quotas held fixed. The learned rule barely changed: the no-compression threshold moved by one grid step (40 → 60 tokens), the chosen rates never changed, and accuracy moved by 0.6 points (0.380 → 0.374). **(b) Fidelity:** synthetic prompts were compared with real prompts of the same category, weighted to the same length mix. In eight of nine category × rate cells the 95% interval excluded zero. Synthetic instruction and summarisation prompts lost 0.02–0.05 more fidelity under compression, and the QA differences changed sign with rate (Figure 10).

![Fidelity difference between synthetic and real prompts by category and rate, with 95% bootstrap intervals](figures/fig8_synthetic_vs_sourced.png)

Synthetic prompts are therefore safe for *calibrating* a selector but are **not a drop-in substitute for real prompts when *measuring* fidelity**: their bias is of the same order as the method differences in Table 8. For this reason the headline method comparisons rest on the 850 real corpus prompts plus the 150 synthetic creative prompts, which are confined to one category, and creative-category results carry this caveat (Long et al., 2024; Liu et al., 2024).

### Robustness and verification

The main results were checked in four ways.

- **Scorer.** Every RQ1 comparison holds under multilingual BERT (LLMLingua-2 beats random deletion by +0.017 to +0.033, all p < 10^−14^), and the mBERT APCS evaluation reaches the same conclusions.
- **Scorer window.** Removing the 2.6% of pairs that exceed AraBERT's 510-word-piece window changes the RQ1 differences by at most 0.002.
- **Replication across data.** The ranking flip, the method ordering, the cost penalty and APCS 1.0.0's accuracy all replicated from pilot to v2 and from test to exam. The size of the cost penalty varies between prompt sets: LLMLingua-2 @ 0.5 adds 10.9% on dev, 19.0% on test and 12.1% on the exam. Its direction never changes.
- **Independent audit.** A script that imports none of the experiment code recomputed every number cited in this chapter from the raw files. It re-joined all 13,438 and 3,828 stored responses to the results tables, recomputed costs from token counts, re-applied the frozen selectors and re-ran the statistical tests. All 117 checks passed. The audit found three documentation errors in the experiment log, which were corrected, and none in the data or results.

### Answering the research questions

| RQ | Answer | Key evidence |
|---|---|---|
| RQ1: effectiveness on Arabic | **Partly effective.** English defaults fail (GPT-2 scoring corrupts Arabic). Classifier-based LLMLingua-2 transfers best, beating random deletion by 0.04–0.05 F1 at every rate, and beats compression-matched LLMLingua. Light compression is tolerable; heavy compression damages most answers and lowers QA correctness from 57% to as little as 32%. Prompt-level evaluation ranks methods backwards. | Table 8; Figures 3 and 4 |
| RQ2: feature effects | **Task type and length matter; morphology does not.** Category explains 6.5% and length 3.4% of unique variance; fragmentation has a small negative effect; morphological density is not significant. QA is the most fragile category once noise is accounted for. Prompt-level features explain only 15% of the variation. | Table 9; Figure 5 |
| RQ3: can a selector choose accurately? | **Yes, with a modest effect.** A tree over task category and length (APCS-v2) beats the best fixed strategy on unseen prompts (+5.8 points, Holm p = 0.028) and the rule-based APCS 1.0.0 (+7.2, p = 0.008). Absolute accuracy (38.5%) remains limited by the noisy, near-uniform labels. A cost-aware variant lowers the bill only slightly (−1.0%) while keeping fidelity high. | Tables 11–13; Figure 8 |
| Additional finding | **Compression can raise the bill.** Under 6:1 output pricing, LLMLingua-2 deletes length instructions, answers grow 1.3–2.4×, and total cost rises 5–24%. Protecting length instructions reverses this (−5 to −8%) and raises fidelity. | Tables 10 and 14; Figures 6 and 9 |

Table: Answers to the research questions

## Chapter Summary

The evaluation first established that a fixed 0.85 threshold could not be used, because a third of identical-prompt answer pairs fail it, and set a noise-derived τ instead. On that basis, LLMLingua-2 is the most effective compressor for Arabic. Output-level evaluation is essential, because prompt-level scores rank methods backwards. Task type and length, not morphology, drive compression outcomes. The original rule-based APCS replicated at about 31% but did not reliably beat the best fixed strategy. The redesigned APCS-v2 did, on pre-registered fresh data. Under output-weighted pricing, compression raised the bill because it deleted length instructions. Protecting those instructions turned the cost penalty into a saving, while a cost-aware selector achieved a small but significant saving at high fidelity. Chapter 6 reflects on these outcomes, their limitations and their implications.
