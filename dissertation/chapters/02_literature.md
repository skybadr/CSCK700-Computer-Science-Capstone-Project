# Background and Review of Literature

## Chapter Introduction

This chapter establishes what is already known about the problem and where the gap lies. Section 2.2 explains why token-based pricing makes Arabic prompts disproportionately expensive. Section 2.3 reviews five bodies of work: prompt compression methods; adaptive, cost-aware LLM use; multilingual tokenisation and morphology; Arabic NLP and LLM evaluation; and the methodology for evaluating compressed prompts. It critically compares the alternatives and closes with the research gap. Section 2.4 sets out the theory on which the APCS is built: compression strategy selection as an algorithm selection problem under a fidelity constraint, with an explicit cost model. Section 2.5 defines the terms used throughout.

## Research Background

Commercial LLM APIs bill each request by the number of tokens processed, with separate prices for input (prompt) tokens and output (completion) tokens. Prices differ by up to two orders of magnitude between providers and models (Chen, Zaharia and Zou, 2024). Output tokens are typically several times more expensive than input tokens; for the model used in this project, the ratio is 6:1 (US$0.20 per million input tokens and US$1.20 per million output tokens). Token count therefore determines cost, latency, energy use and how much of the model's context window a request uses (Wan et al., 2024). Inference energy grows with the number of tokens processed and generated (Samsi et al., 2023), so shorter exchanges are also greener.

Tokenisers do not treat languages equally. Ahia et al. (2023) measured OpenAI's API on 22 typologically diverse languages and found that some languages need up to five times more tokens than English for the same content, so speakers of those languages pay more while often receiving worse results. Petrov et al. (2023) found differences of up to fifteen times between language pairs, and showed that even byte-level tokenisers leave large gaps. Arabic is among the disadvantaged languages. In this project the tokeniser of the target API (cl100k_base) produced a mean of 4.1 tokens per Arabic word across the 1,000 AraPromptBench prompts (10th–90th percentile 3.7–4.6). In other words, a typical Arabic word is split into four pieces, each billed separately. Arabic users therefore pay what can fairly be called a *tokeniser tax*.

Two responses to the tax exist. Model builders can extend the vocabulary with Arabic tokens: ALLaM merged a dedicated Arabic tokeniser into an English model's vocabulary (Bari et al., 2025), and AraLLaMA expanded the vocabulary progressively during training (Zhu et al., 2025). Both remedies are available only to whoever trains the model. Developers who consume a commercial API cannot change its tokeniser; they can only change what they send. For them, prompt compression is the practical lever, and that is the setting this project addresses.

Arabic LLM capability has improved quickly. Arabic-centric models such as AceGPT (Huang et al., 2024a) and ALLaM, and benchmarks such as ORCA (Elmadany, Nagoudi and Abdul-Mageed, 2023), Dolphin (Nagoudi et al., 2023) and ArabicMMLU (Koto et al., 2024), have made Arabic understanding and generation measurable. Efficiency for Arabic, however, has received far less attention than capability. That imbalance is the background to the research questions.

## Literature Review

### Prompt compression methods

Li et al. (2025) survey the field and divide prompt compression into two families. **Soft-prompt** methods compress a prompt into learned continuous vectors. Gisting trains a model to summarise an instruction into a few "gist" tokens, reaching up to 26-fold compression (Mu, Li and Goodman, 2023). AutoCompressors recursively summarise long documents into summary vectors (Chevalier et al., 2023). The In-context Autoencoder encodes a context into compact memory slots using a lightly fine-tuned copy of the LLM (Ge et al., 2024). These methods compress very effectively but need access to the model's weights or embedding layer. A developer calling a closed commercial API cannot send vectors instead of text, so soft-prompt methods are unusable in the setting of this project.

**Hard-prompt** methods instead output a shorter natural-language prompt that any API accepts. Selective Context removes phrases and sentences with low self-information, estimated by a small causal language model. It halves context length with only a 0.023 drop in BERTScore on English tasks (Li et al., 2023). LLMLingua (Jiang et al., 2023) refines the same idea with a coarse-to-fine procedure: a budget controller allocates compression across prompt sections, then tokens are removed iteratively according to a small model's perplexity. It reaches up to 20-fold compression on English reasoning and conversation benchmarks. LongLLMLingua adds question-aware scoring and document reordering for long retrieval contexts (Jiang et al., 2024). RECOMP compresses retrieved documents into extractive or abstractive summaries before they are added to the prompt (Xu, Shi and Choi, 2024). Nano-Capsulator rewrites prompts into shorter natural-language capsules with a fine-tuned LLM (Chuang et al., 2024). Other recent work learns which tokens to delete with reinforcement learning (Jung and Kim, 2024), prunes in-context examples for reasoning tasks (Huang et al., 2024b), or scores whole sentences with a context-aware encoder (Liskavets et al., 2025).

LLMLingua-2 (Pan et al., 2024) departs from perplexity scoring. Its authors argue that the entropy of a causal model is unidirectional and not aligned with the compression objective. They distil keep-or-drop labels from GPT-4 and train a bidirectional token classifier (XLM-RoBERTa-large) to predict them. The result is three to six times faster than LLMLingua and more faithful to the original prompt. Two properties make LLMLingua-2 especially relevant here: its encoder is multilingual, and its compression rate can be controlled precisely.

Evaluations of these methods are becoming more critical. Jha et al. (2024) find that simple extractive compression often beats token pruning, despite claims to the contrary. Łajewska et al. (2025) show that several state-of-the-art methods fail to preserve key details such as entities, which hurts complex tasks. Nagle et al. (2024) derive the distortion–rate function for black-box prompt compression and show a large gap between current methods and the optimum. The gap is narrowed by compression that is *query-aware* and *variable-rate*, meaning it knows the downstream task and adapts the rate to each input. This is the theoretical case for choosing the compression rate per prompt rather than fixing it globally. Table 1 summarises the alternatives against the requirements of this project.

| Method family | Representative work | Black-box API compatible | Multilingual scorer available | Rate control | Suitability here |
|---|---|---|---|---|---|
| Soft prompts / memory slots | Mu, Li and Goodman (2023); Chevalier et al. (2023); Ge et al. (2024) | No | Model-dependent | Fixed by training | Excluded: needs model internals |
| Perplexity-based token deletion | Li et al. (2023); Jiang et al. (2023) | Yes | Yes, if the scorer is swapped | Approximate | Included (LLMLingua) |
| Question-aware long-context | Jiang et al. (2024); Xu, Shi and Choi (2024) | Yes | Partly | Approximate | Excluded: targets long retrieval contexts |
| Classifier-based token deletion | Pan et al. (2024) | Yes | Yes (XLM-RoBERTa) | Precise | Included (LLMLingua-2) |
| Generative rewriting | Chuang et al. (2024) | Yes | Needs a fine-tuned LLM | Weak | Excluded: needs training, adds an LLM call |
| Learned or sentence-level selection | Jung and Kim (2024); Liskavets et al. (2025) | Yes | Needs retraining for Arabic | Coarse (sentence) or learned | Excluded: no Arabic models; coarse units suit long contexts |
| Random token deletion | — (baseline) | Yes | Not applicable | Precise | Included as control |

Table: Prompt compression alternatives evaluated against the project's requirements

Every one of these methods was developed and evaluated on English. LLMLingua's reference scorers are GPT-2 and LLaMA-7B. LLMLingua-2's distillation data (MeetingBank) is English. None of the papers reports results on Arabic or any other morphologically rich language. Whether English-learned notions of token importance transfer to Arabic is therefore untested. RQ1 addresses this.

### Adaptive and cost-aware LLM use

A second body of work shows that choosing *per query* beats committing to one global strategy. FrugalGPT combines prompt adaptation, model approximation and an LLM cascade. It matches GPT-4's accuracy at up to 98% lower cost by sending each query only as far up the cascade as needed (Chen, Zaharia and Zou, 2024). Hybrid LLM trains a router that predicts query difficulty and sends easy queries to a small model, cutting large-model calls by up to 40% without loss of quality (Ding et al., 2024). RouteLLM learns routers from human preference data and more than halves cost (Ong et al., 2025). FORC uses a meta-model to predict each candidate model's performance on an input and chooses the cheapest adequate one, reducing cost by 63% (Šakota, Peyrard and West, 2024).

These systems share the APCS's structure: inexpensive features of the input drive a choice among options with different cost–quality trade-offs. They differ in what they choose, which is models rather than compression strategies, and all were evaluated on English. They also optimise expected *cost* directly, using measured prices. As Chapter 5 shows, that turns out to matter: a selector that optimises token reduction is not optimising the bill.

The same structure has a long history in computer science as the **algorithm selection problem** (Rice, 1976). Given a problem space, a set of algorithms and a performance measure, a selection mapping uses features of each problem instance to choose the algorithm expected to perform best. Framing compression choice this way identifies the components the APCS needs (instance features, a candidate set, a performance measure and a learned mapping) and the right baseline: the *single best solver*, the one fixed algorithm that performs best on average.

### Multilingual tokenisation and morphology

Why might Arabic behave differently under compression? The tokenisation literature offers mechanisms. Limisiewicz, Balhar and Mareček (2023) show that multilingual tokenisers allocate vocabulary unevenly across languages, and that the allocation predicts downstream performance gaps. Ali et al. (2024) find that tokeniser choice measurably changes both downstream accuracy and training cost. English-centric tokenisers used for multilingual models cause severe downstream degradation and up to 68% additional training cost. On morphology, Park et al. (2021) find that languages with richer inflectional morphology are harder to model, though the effect weakens when segmentation respects morpheme boundaries. Arnett and Bergen (2025) test three explanations for the gap. They find no evidence that tokenisers' alignment with morpheme boundaries explains it, some evidence for tokenisation quality, and the clearest effect from training-data size. The gap largely disappears when datasets are equated in size after adjusting for each script's encoding efficiency (its "byte premium"), so morphological type itself confers no intrinsic disadvantage. MorphBPE builds morpheme boundaries into byte-pair encoding and improves training efficiency for morphologically rich languages, including Arabic (Asgari et al., 2026).

Two hypotheses follow for compression. First, if a prompt's tokens are mostly fragments of words, deleting tokens may break words. Deleting one byte-level fragment of a multi-byte Arabic character can even corrupt the text. Second, prompts with dense morphology, where many function words are fused into each orthographic word, may carry more information per word and so tolerate deletion less. The SDR turned the second hypothesis into a morphology-based selection rule. RQ2 tests both. The evidence above suggests morphology itself may matter less than tokenisation and the amount of text, and that is the outcome Chapter 5 reports.

### Arabic NLP and LLM evaluation

Habash (2010) describes the properties of Arabic that complicate processing: root-and-pattern morphology, clitic attachment, optional diacritics, and the coexistence of Modern Standard Arabic with regional dialects. Language-specific processing pays off. The Farasa segmenter splits words into clitics and stems quickly and accurately (Abdelali et al., 2016). AraBERT, pre-trained on Arabic with segmentation-aware preprocessing, outperforms multilingual BERT on Arabic understanding tasks (Antoun, Baly and Hajj, 2020). Both are used in this project: Farasa to measure morphological density, and AraBERT as the fidelity scorer.

Arabic LLM evaluation has matured rapidly. GPTAraEval evaluated ChatGPT on 44 Arabic tasks and found it clearly behind smaller fine-tuned models on many of them (Khondaker et al., 2023). ArabicMMLU measures knowledge with school-exam questions from Arabic-speaking countries (Koto et al., 2024). ORCA and Dolphin provide understanding and generation benchmarks respectively (Elmadany, Nagoudi and Abdul-Mageed, 2023; Nagoudi et al., 2023). Public instruction corpora now exist: CIDAR provides 10,000 culturally reviewed instruction–response pairs (Alyafeai et al., 2024), and the Aya dataset provides human-written instructions in many languages and dialects (Singh et al., 2024). Together with older task corpora (XL-Sum for summarisation (Hasan et al., 2021), the Essex Arabic Summaries Corpus (El-Haj, Kruschwitz and Fox, 2010), TyDi QA (Clark et al., 2020) and ARCD (Mozannar et al., 2019) for question answering), these made it possible to build AraPromptBench from credible, licensed sources rather than from invented prompts.

None of these benchmarks studies efficiency. They measure whether a model *can* do a task in Arabic, not what it costs or how far a prompt can be shortened. No Arabic prompt compression benchmark existed when this project began.

### Evaluating compressed prompts

Three methodological issues from the literature shape how compression should be evaluated.

**Fidelity measurement.** BERTScore compares texts by matching contextual token embeddings, which captures paraphrase better than n-gram overlap (Zhang et al., 2020). Its behaviour depends on the encoder, so an Arabic-specific encoder may separate good from bad Arabic outputs better than multilingual BERT (Devlin et al., 2019). Experiment 01 tests this (Chapter 4). An alternative is to ask a strong LLM to judge answer quality. Such judges agree well with human preferences (Zheng et al., 2023; Chiang and Lee, 2023), but they add an extra paid call per comparison and bring biases of their own, including a preference for longer answers. For a benchmark of tens of thousands of comparisons, a deterministic embedding-based metric is more reproducible and affordable. It also matters *which* texts are compared. Comparing the compressed prompt with the original rewards keeping words, not keeping meaning. Comparing the LLM's answers to the two prompts measures what users actually experience. Selective Context and LLMLingua both report task performance on outputs for this reason (Li et al., 2023; Jiang et al., 2023).

**Non-determinism.** Supposedly deterministic LLM settings are not deterministic. Ouyang et al. (2025) found that ChatGPT gives different code for the same prompt even at temperature zero. Song et al. (2025) show that evaluations based on a single sample per prompt hide considerable variability. A fidelity threshold must therefore be judged against the model's own noise: if two calls with the *identical* prompt often score below a threshold, that threshold cannot separate compression damage from chance. The SDR's fixed threshold of 0.85 was not tested this way. This project tests it (Chapter 5).

**Output length.** LLMs trained with reinforcement learning from human feedback tend to produce longer answers, because preference models reward length (Singhal et al., 2024). Automatic evaluators share the bias and must be corrected for length (Dubois et al., 2024). Explicit length instructions are one of the main ways to control answer length (Jie et al., 2024), and adding a token budget to the prompt cuts reasoning tokens substantially (Han et al., 2025). The implication for compression has not been drawn in the compression literature: when output costs more than input, the words that control *how much* the model writes may be the most valuable words in the prompt.

Two further concerns apply to evaluation data. **Contamination**: benchmark items seen during development, or during model training, inflate measured performance (Sainz et al., 2023; Deng et al., 2024). Public test sets can be traced in a model's outputs (Golchin and Surdeanu, 2024). Selectors tuned after looking at test results can be contaminated in the same way, so a design changed in response to test results must be re-evaluated on unseen data. **Synthetic data**: LLM-generated examples can fill gaps cheaply but may differ systematically from real data (Long et al., 2024; Liu et al., 2024), so their effect must be measured rather than assumed.

### The research gap

The literature establishes that prompt compression can reduce token use substantially (Jiang et al., 2023; Pan et al., 2024); that adaptive per-input selection beats fixed strategies for cost-efficient LLM use (Chen, Zaharia and Zou, 2024; Šakota, Peyrard and West, 2024); and that tokenisation imposes a measurable cost penalty on Arabic (Ahia et al., 2023; Petrov et al., 2023). These findings have not been connected. Specifically:

1. no published study evaluates prompt compression methods on Arabic, so it is unknown whether English-trained compressors transfer (RQ1);
2. no study relates prompt features such as length, task type, tokeniser fragmentation or morphological density to compression outcomes in Arabic (RQ2);
3. no feature-based recommender exists for selecting a compression strategy per prompt, in Arabic or any other language (RQ3);
4. compression is evaluated by token reduction or input cost. Its effect on *output* length, and so on the total bill under output-weighted pricing, has not been measured.

The fourth gap emerged during the project rather than from the initial review. It is included because the literature on length bias (Singhal et al., 2024) and length control (Jie et al., 2024) predicts it, and because the results confirm it.

## Theory

**Compression as constrained optimisation.** Let *p* be a prompt and *M* the LLM under test. A compression strategy *s* = (method, target rate) maps *p* to a shorter prompt *s*(*p*). Its token compression ratio is TCR(*s*, *p*) = 1 − |*s*(*p*)| / |*p*|. Its fidelity is *F*(*s*, *p*) = BERTScore F1 between *M*(*s*(*p*)) and *M*(*p*), the answers to the compressed and original prompts. The proposal defines the best strategy as the one that "maximises token reduction while maintaining semantic fidelity above a defined threshold". Formally, for a candidate set *S* that always contains *no compression* (TCR = 0, *F* = 1):

$$ *s*∗(*p*) = arg max over *s* ∈ *S* of TCR(*s*, *p*),  subject to  *F*(*s*, *p*) ≥ τ

This *best-balance* label exists for every prompt, because no compression always satisfies the constraint. It turns per-prompt evaluation into a classification target.

**Selection as algorithm selection.** Following Rice (1976), the APCS is a mapping from a feature vector *f*(*p*) to a strategy in *S*, and is judged by how often it chooses *s*∗(*p*) on unseen prompts. The natural baselines are the single best fixed strategy, random selection and never compressing. Interpretable mappings, such as ordered threshold rules and shallow decision trees, are preferred over black-box classifiers. A developer can inspect them, they can be audited against the evidence that produced them, and they suit small calibration sets. This follows the SDR's emphasis on explainability.

**The cost model.** For a call with *n*~in~ prompt tokens and *n*~out~ completion tokens, the cost is *C* = π~in~·*n*~in~ + π~out~·*n*~out~, with π~out~ = 6π~in~ for the model studied. Compression reduces *n*~in~ by the TCR, but the saving only materialises if *n*~out~ does not grow. The total saving from compressing is π~in~·Δ*n*~in~ − π~out~·Δ*n*~out~, so a 10% increase in answer length cancels a 60% reduction in prompt length when answers and prompts are of similar size. Token reduction is therefore a reliable proxy for cost only if compression leaves answer length unchanged. This theory makes the fourth research gap testable. It is also why the project measures cost from the token counts the API actually bills, never from TCR.

**Measurement noise and the threshold.** Because *M* is not deterministic, *F* has a noise floor: the distribution of BERTScore F1 between two answers to the *identical* prompt. A sensible τ should be failed by pure noise only rarely. This project sets τ at the 0.4th percentile of the repeat-call distribution (99.6% specificity), rounded to a convenient value. A compressed prompt that falls below τ has then almost certainly been damaged by compression rather than by chance.

## Terminology

- **Prompt compression:** reducing a prompt's token count before sending it to an LLM while trying to preserve the information it needs.
- **Target rate (keep rate):** the fraction of tokens a compressor is asked to keep (0.7, 0.5 or 0.3 in this project). *Achieved keep* is the fraction actually kept.
- **Token Compression Ratio (TCR):** 1 − compressed tokens / original tokens, counted with the cl100k_base tokeniser.
- **Fragmentation ratio:** tokens per orthographic word; the size of the tokeniser tax for a given prompt.
- **Morphological density:** mean number of Farasa segments (clitics and stems) per Arabic word.
- **Prompt-level fidelity:** BERTScore F1 between the compressed and original prompts. Diagnostic only.
- **Output-level fidelity:** BERTScore F1 between the LLM's answers to the compressed and original prompts. The primary fidelity measure.
- **Noise ceiling:** the distribution of output-level F1 between two answers to the identical prompt.
- **τ (tau):** the fidelity threshold derived from the noise ceiling (0.65 for AraBERT).
- **Best-balance label:** for a prompt, the strategy with the largest TCR whose output F1 ≥ τ, or no compression if none qualifies.
- **Cheapest-faithful label:** for a prompt, the strategy with the lowest measured API cost whose output F1 ≥ τ.
- **Length instruction:** an explicit constraint on answer length, such as "briefly", "in three sentences" or "no more than fifty words".
- **LLMLingua-2-protected:** the variant developed in this project that keeps sentences containing length instructions verbatim and compresses the rest.
- **APCS 1.0.0, APCS-v2, APCS-cost:** the calibrated rule-based selector, the redesigned decision-tree selector, and the cost-aware selector (Chapter 4).
- **Dev, test and exam sets:** the 800-prompt development split, the 200-prompt held-out split and the 400-prompt fresh exam set.

## Chapter Summary

The literature shows that hard-prompt compression is the only family usable with a closed API; that LLMLingua and LLMLingua-2 represent its two main scoring philosophies; that adaptive per-input selection outperforms fixed strategies elsewhere in cost-efficient LLM use; and that tokenisation makes Arabic disproportionately expensive. It has not tested compression on Arabic, related prompt features to compression outcomes, built a compression selector, or measured compression's effect on output length and total cost. The theory section formalised the best-balance target, framed the APCS as an algorithm selector and showed with a cost model why token reduction need not reduce the bill. Chapter 3 turns this into a specification and research design.
