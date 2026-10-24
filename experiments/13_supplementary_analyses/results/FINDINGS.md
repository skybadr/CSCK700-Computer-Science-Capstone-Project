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

## Part 2 — checks raised by an independent pre-submission review (`review_checks.py`)

Computed 2026-10-24 from stored results (no API calls; no selector re-tuned; the
exam set was already used, so every exam figure below is exploratory, not
pre-registered). Output: `results/review_checks.json`.

- **L. Billing tokeniser:** billed input tokens = o200k_base count + 6 (message
  overhead) for all 1,000 v2 prompts (r = 1.000), so gpt-5.6-luna bills with
  o200k_base, not cl100k_base. Tokens per Arabic word: cl100k 4.12 (p10–p90
  3.67–4.57), o200k 1.91 (1.63–2.19). TCR of 2,400 dev LLMLingua-2 variants under
  the two tokenisers: r = 0.975, means 0.512 vs 0.524 — cl100k is a sound fixed
  measurement unit; costs were always from billed counts.
- **M. H1 by category (exam):** APCS-v2 vs always LL2@0.5 = +5.75 pts (62 vs 39,
  p = 0.028). Without creative (n = 340): +2.35 pts (39 vs 31, p = 0.40); without
  summarisation: +3.0 (p = 0.31). Net correct-prompt gain: creative +15,
  summarisation +14, instruction +2, QA −8.
- **N. H3 on the total bill:** total −0.97%, 95% ratio-of-sums bootstrap CI
  [−2.94%, +1.30%]; 229/400 per-prompt differences exactly zero (APCS-cost picks
  "none" for 57%); one-sided sign-flip permutation test on the mean p = 0.19.
  The pre-registered Wilcoxon criterion is met, but the total saving is not
  distinguishable from zero.
- **O. τ sensitivity, frozen selectors:** with exam labels re-derived at τ = 0.60
  APCS-v2 still beats always @0.5 (+7.0, p = 0.003); at 0.70 (+1.0, p = 0.75) and
  with per-category τ (Table 11 percentiles; −1.25, p = 0.68) it does not. H2
  holds at 0.65 and per-category τ, not at 0.60 or 0.70.
- **O2. τ sensitivity, design repeated on dev at each τ** (same CV procedure as
  `design_selectors.py`; comparator = best fixed strategy on dev at that τ):
  tree vs best fixed: 0.60 −1.75 (always @0.3 53.0% vs tree 51.3%, p = 0.55);
  0.65 +5.75 (p = 0.028); 0.70 +16.0 (p < 0.001); per-category +9.5 (p = 0.001).
  Tree vs token-count rule: +8.0 to +11.75 at every τ (p ≤ 0.001). The approach
  is robust to τ; a shipped tree is tied to the τ it was trained for.
- **P. RQ2 with the noise ceiling as covariate (dev, LL2 @0.5):** adding each
  prompt's repeat-call F1 raises R² 0.148 → 0.159; unique R² of category falls
  6.5 → 2.1 pts, length 3.7 pts; category effects stay significant (all p < 0.01
  vs creative); morphology β = +0.001 (p = 0.79); fragmentation β = −0.011
  (p = 0.015).
- **Q. Per-category lookup (exploratory):** best dev rate per category
  (instruction @0.3, summarisation @0.3, QA @0.5, creative @0.7) scores 38.25% on
  the exam vs APCS-v2 38.5% (p = 1.0) and vs always @0.5 32.75% (p = 0.14). The
  APCS-v2 gain is the task-category signal, which a lookup captures equally well.
- **R. Length-marker precision:** 625/1,400 prompts flagged; 154 only by كلمة, of
  which 53 have no number before it (e.g. "ما إعراب كلمة", "كلمة مرور") → ≈8.5%
  likely false positives. Known misses: باختصار, مختصر, hamza-less بايجاز, لا تتجاوز.
- **S. APCS-cost on exam QA:** compresses 73/100 (45 LL2@0.5, 28 protected@0.3);
  QA correctness 42% vs 57% uncompressed — deployment rule 3 must override it.
- APCS-v2 closes (38.5 − 32.75)/(100 − 32.75) = 8.6% of the gap between the best
  fixed strategy and a perfect selector.

Also checked and rejected: clitic-tolerant QA matching (stripping و/ف/ب/ل/ك/ال
before matching) changes uncompressed exam QA correctness from 57/100 to only 58/100.
