# Experiment 04 — Findings: output-level evaluation (gpt-4o-mini)

**Run:** 2026-07-05. 4,173 unique calls to gpt-4o-mini (temp 0.0, max_tokens 512),
covering all 5,000 Experiment 03 rows (identical texts share one call).
Actual usage: 277,222 prompt + 1,024,005 completion tokens = **$0.656**
(`api_usage.json`). 401 responses truncated at the 512 cap (256 creative,
126 instruction, 12 qa, 7 summ) — see caveats. Data: `output_eval_results.csv`,
raw responses in `responses.jsonl`.

Output-level fidelity = BERTScore F1 (AraBERT primary / mBERT sensitivity)
between the response to the compressed prompt and the response to the original.

## Finding 1 — The method ranking FLIPS at output level (headline result)

Paired Wilcoxon, out-F1-AraBERT, llmlingua2 vs random_deletion:

| rate | llmlingua2 | random | diff | p | prompt-level said |
|---|---|---|---|---|---|
| 0.3 | 0.613 | 0.571 | **+0.042** | 2.9e−22 | random better (−0.022, p=1.4e−22) |
| 0.5 | 0.672 | 0.645 | **+0.027** | 2.2e−12 | random better (−0.036, p=1.0e−44) |
| 0.7 | 0.735 | 0.712 | **+0.023** | 4.2e−09 | random better (−0.042, p=5.8e−58) |

Prompt-level similarity and output-level fidelity give *opposite*, both highly
significant, method rankings. Intelligent compression preserves what matters to
the LLM even while looking less similar as text. This vindicates the two-level
evaluation design and is a core methodological contribution: **Arabic
compression studies that rank methods on prompt-level similarity would select
the wrong method.** (Overall Pearson r between the two levels is 0.855 —
positively correlated in general, yet systematically biased between methods.)

## Finding 2 — Like-for-like, LLMLingua-2 also beats LLMLingua-1

LLMLingua-1 (Qwen scorer) posts high aggregate F1 (0.836–0.873) but only
because it barely compresses (keep 0.79–0.91; ≈1.0 on qa/creative). On the
610 rows where it *did* compress (mean keep 0.63): out-F1 = 0.662. LLMLingua-2
at comparable compression (keep 0.68, rate 0.7) on the *same prompts*:
out-F1 = **0.760**. LLMLingua-2 dominates the perplexity-based method
like-for-like, consistent with LLMLingua-2's own paper but now shown for Arabic.

## Finding 3 — Task categories differ in compression sensitivity (APCS rationale)

Out-F1-AraBERT at rate 0.5 (llmlingua2): summarisation 0.714 > instruction
0.692 > qa 0.652 > creative 0.629. Summarisation prompts (long, redundant
passages) tolerate compression best; creative prompts are most fragile (short,
and any nudge changes the generated story). This is the empirical seed for
per-category APCS rules.

## Finding 4 — The 0.85 threshold is strict even at output level

% rows with out-F1 ≥ 0.85: llmlingua2 0/1.2/7.0% at rates 0.3/0.5/0.7; random
0.2/0.2/4.0%. (LLMLingua-1's 56–59% again reflects non-compression.) Two
readings, both reportable: (a) genuine result — Arabic prompt compression
causes substantial response drift at these rates, i.e. compression is *costly*
for Arabic; (b) calibration point — as with prompt-level (Exp 01), a fixed
0.85 needs interpretation relative to a realistic ceiling. A same-prompt
repeat-call baseline (what does F1 look like for two calls with the *identical*
original prompt?) should be measured before final threshold-based claims;
flagged for the final 1,000-prompt protocol (cheap: 500 extra calls).

## Caveats

- 401/4,173 responses truncated at max_tokens 512, concentrated in creative
  (256) — truncation depresses similarity for long generations; category
  comparisons involving creative should note this.
- Temperature 0.0 + response reuse for identical texts makes noop F1 = 1.0 by
  construction; the repeat-call ceiling above is the proper noise floor.
- Single LLM (gpt-4o-mini); cross-model replication is a desirable extension
  (SDR lists claude-haiku as candidate).

## Status

RQ1's evidence base is now complete for the pilot: TCR (Exp 03) × output-level
fidelity (Exp 04) per prompt/method/rate enables Pareto-frontier labelling —
the input for feature analysis (RQ2) and APCS rule derivation (RQ3).
