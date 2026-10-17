# Experiment 13 — Findings (supplementary analyses)

- **A. Clitic fragmentation (cl100k):** وبالكتاب ("and with the book", 4
  morphemes) = 6 tokens, split letter by letter (و|ب|ال|ك|ت|اب); English "and
  with the book" = 4 tokens. فسيكتبونها ("so they will write it") = 10 tokens vs 5.
  cl100k keeps whole Arabic letters; GPT-2's byte-level BPE splits the 2-byte
  letter ك into raw bytes (<d9>|<83>), the mechanism behind Exp 02's corruption.
- **B. Scorer tokenisers (tokens per Arabic word, 200 dev prompts):** cl100k
  4.13; GPT-2 5.86; Llama-2-7B 5.32; Qwen2.5/Qwen3 2.27; Llama-3.1/3.2 2.28;
  AraGPT2 1.56; Jais 1.46; XLM-R large 1.81; mBERT 2.26.
- **C. Noise ceiling by category (dev):** identical repeats / median F1 / % below
  0.85: QA 66.5% / 1.000 / 5.0%; summarisation 20.0% / 0.947 / 9.5%; instruction
  22.1% / 0.881 / 39.3%; creative 1.7% / 0.743 / 95.8%. τ = Q0.004 = 0.640 → 0.65.
- **D. QA (exam):** when the gold answer survives compression, answers are correct
  71.5% of the time; when deleted, 17.1% (1,200 compressed QA variants).
  Gold-answer retention: LLMLingua-2 @0.7/0.5/0.3 55/35/23%; random deletion
  31/16/10%; LLMLingua 66/64/60%.
- **E. Morphology (dev):** mean 1.69 segments/word, SD 0.18 (CV 10.5%); Spearman
  0.59 with fragmentation; VIFs 1.35–1.54 (no harmful collinearity). Little
  variation, and what there is overlaps with fragmentation.
- **F. Cost break-even (dev):** mean billed input 109, output 127 tokens. Answer
  growth that cancels the input saving: 4.3/7.0/9.7% at @0.7/0.5/0.3; observed
  total output growth 9.5/19.5/37.4%.
- **G. APCS-v2 on exam QA:** the leaf "QA, >433 tokens → @0.7" was right 1/16
  (6%) vs fixed @0.5 7/16 (44%); "short, highly structured → @0.7" 21% vs 36%.
- **H. BERTScore and length:** |log answer-length ratio| vs output F1 (LL2 @0.5):
  Spearman −0.47; F1 0.634 when the answer grew >50% (36% of prompts) vs 0.712.
- **I. Precision:** test (n=200) accuracy CI ±6.4 pts, minimum detectable paired
  difference at 80% power 11.1 pts; exam (n=400) ±4.7 pts, 7.0 pts.
- **K. Length regex:** matched 625 of 1,400 v2+exam prompts; unmatched variants
  present: في حدود (10 prompts), لا تتجاوز (1).
