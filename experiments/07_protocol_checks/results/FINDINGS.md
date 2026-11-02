# Experiment 07 — Findings: protocol checks

**Run:** 2026-07-09. (a) 500 repeat calls to gpt-4o-mini, $0.087;
(b) 20-prompt LLMLingua-1 probe, local; (c) AraBERT window audit, local.

## Finding A1 — gpt-4o-mini at temperature 0 is far from deterministic

Sending the *identical* prompt twice: only **13.0%** of response pairs are
string-identical. Repeat-pair F1-AraBERT: mean 0.894, median 0.901,
p25 0.836, p5 0.744, min 0.676. This is the noise ceiling for all
output-level fidelity claims.

## Finding A2 — The SDR's F1 ≥ 0.85 threshold is untenable at output level

**31.2% of identical-prompt repeat pairs fail 0.85.** A threshold that rejects
the LLM's own run-to-run noise a third of the time cannot define "acceptable
compression". Failure rates at other thresholds: 16.0% @ 0.80, 5.8% @ 0.75,
**0.4% @ 0.70**. The pilot's provisional τ = 0.70 (Exp 06) is thereby
retroactively validated as a ≈99.6%-specificity criterion: a compressed
prompt failing τ = 0.70 is almost certainly genuinely damaged, not unlucky.
For the final protocol: keep τ = 0.70 globally, or use category ceilings.

## Finding A3 — Ceiling-relative fidelity REORDERS category fragility

Ceilings differ sharply by category (median repeat-pair F1): summarisation
0.964, qa 0.949, instruction 0.902, creative 0.802. Comparing llmlingua2@0.5
against each category's own ceiling:

| Category | ceiling median | F1 @0.5 | retained % of ceiling |
|---|---|---|---|
| creative | 0.802 | 0.629 | **78.5%** (least damaged) |
| instruction | 0.902 | 0.692 | 76.7% |
| summarisation | 0.964 | 0.714 | 74.1% |
| qa | 0.949 | 0.652 | **68.7%** (most damaged) |

Exp 04's absolute ranking ("creative most fragile") inverts: creative outputs
are intrinsically variable (low ceiling), while **QA suffers the most
compression-specific damage** — consistent with qualitative evidence
(compression deletes task-critical details like numbers). Dissertation
implication: report output fidelity both absolute and ceiling-normalised;
the ceiling-normalised view is the fairer basis for category claims and for
final APCS rule calibration.

## Finding B — LLMLingua-1's length gate is intrinsic (target_token ≠ escape)

Driving compression by absolute budget (`target_token = rate × scorer-token
count`) instead of `rate` produces **identical achieved keep-rates in every
cell** (qa/creative 0.99–1.00; instruction 0.71–0.80; summarisation 0.54–0.65;
0 errors). The gate is algorithmic, not a parameter artefact. The benchmark's
treatment of LLMLingua-1 (as-shipped behaviour, documented) is fair and the
"perplexity-based compression is inoperative on short Arabic prompts" claim
stands against the obvious reviewer objection.

## Finding C — AraBERT's 512-wordpiece window is a non-issue

Only **0.6%** of Exp 04 response pairs have either side exceeding 510 AraBERT
wordpieces (26/4,500; 25 in instruction), and affected pairs score no
differently (0.666 vs 0.667 mean F1). Arabic responses tokenise *compactly*
under AraBERT (median 176 wordpieces for responses capped at 512 GPT tokens),
because its Arabic-optimised vocabulary ≈3× denser than cl100k. No protocol
change needed; documented as a verified non-threat.

## Consequences for the final (~1,000-prompt) protocol

1. Fidelity threshold: τ = 0.70 global (0.4% noise-failure) — or
   category-normalised scores with a single retained-fraction criterion.
2. Report ceiling-normalised fidelity alongside absolute; re-measure the
   ceiling on the final dataset (500–1,000 extra calls, ≈$0.10–0.20).
3. The SDR's 0.85 threshold is retired for output-level use, with Finding A2
   as the documented justification (it remains fine as a *prompt-level*
   diagnostic bound in Exp 01's construct-validity sense).
4. Exp 04's design choice of reusing one response per unique text (making
   noop pairs score exactly 1.0) is retained but must be stated: the noop
   anchor is noise-free by construction; all compressed variants carry one
   draw of API noise, bounded by this ceiling.
