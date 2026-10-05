# ABSTRACT

LLM APIs charge per token, and their tokenisers split Arabic words into far more tokens than English ones, so Arabic users pay more for the same content. Prompt compression can cut this cost, but existing methods were built and tested on English. This dissertation asks how well they work on Arabic, which prompt features drive their performance, and whether a recommender can choose the best strategy for each prompt.

Following design science, the project built AraPromptBench (1,000 Arabic prompts from six public corpora plus synthetic creative prompts) and a 400-prompt fresh exam set. It compared LLMLingua, LLMLingua-2 and random deletion at three rates over more than 17,000 API calls. Fidelity was measured on the LLM's answers with AraBERT BERTScore, against a threshold derived from the model's own repeat-call noise.

LLMLingua-2 was the best compressor, beating random deletion by 0.04–0.05 F1 at every rate. Prompt-level similarity ranked the methods backwards. Task type and length explained outcomes; morphology did not. The artefact, the Arabic-Aware Prompt Compression Selector (APCS), reached 31% accuracy as a rule-based selector. A decision-tree redesign reached 38.5% on the pre-registered exam, significantly above the best fixed strategy (Holm p = 0.028). Unexpectedly, compression raised the total bill by 5–24%, because it deleted answer-length instructions and answers grew. Protecting those instructions turned the penalty into a 5–8% saving. Arabic prompt compression must therefore be judged on outputs and total cost, not tokens saved.

# ACKNOWLEDGEMENTS

*[Author to complete: thanks to the Dissertation Advisor, Dr Laud Charles Ochei, and the Dissertation Lead, Dr Andrea Corradini; anyone else to acknowledge.]*

This project uses publicly available datasets released by their authors: CIDAR, the Aya dataset, XL-Sum, the Essex Arabic Summaries Corpus, TyDi QA and ARCD. Their licences are recorded per prompt in AraPromptBench.

**Use of generative AI.** *[Author to verify against the University's and the module's policy on generative AI and edit as required.]* Generative AI tools were used in this project in the following ways: (i) as a coding assistant for the experiment pipeline, analysis scripts and the APCS package, all of which the author reviewed, ran and tested; (ii) to generate the synthetic creative-writing prompts and the synthetic probe prompts, which are identified as synthetic throughout and whose effect was measured (Experiment 09); and (iii) to assist with drafting and editing this dissertation. All research decisions, interpretations and conclusions are the author's own. Every reference was checked against its published source, and every reported number was verified against the raw results by an independent audit script.
