# Experiment 10 — Final benchmark on AraPromptBench v2 (+ probe pool)

The definitive measurement run: every headline dissertation number comes from
here. It applies all protocol fixes flagged in Exps 04–08 and closes the
evaluation items promised in the proposal/SDR that the pilot did not cover.

## Protocol vs pilot

| Aspect | Pilot | Final | Why |
|---|---|---|---|
| LLM under test | gpt-4o-mini | **gpt-5.6-luna** (exact snapshot stored on every response) | 4o family retired; Luna is the newest-generation budget tier, which fits the thesis's cost-sensitive framing |
| Output cap | 512 max_tokens | 1,024 max_completion_tokens | pilot creative truncation (Exp 04) |
| Noise ceiling | measured post hoc (Exp 07) | measured in-run: one repeat call per original prompt | τ must come from THIS model's noise floor |
| τ | 0.70 | derived from the dev ceiling at 99.6% specificity, rounded to 0.05; separately for AraBERT and mBERT | Exp 07 Finding A2 |
| Dataset | 500 pilot, unsplit | v2: 800 dev / 200 test (test read only by `apcs_final.py`) | contamination-free headline claims |
| Probe pool | — | 456 synthetic prompts measured in the SAME run | Exp 09 deconfound |
| Morphology feature | punctuation-inflated | corrected (`apcs.features.density_from_segmented`) | Exp 05 erratum |

**Run history.** A first API run on gpt-5.4-mini (July 2026) stopped at
6,799/13,438 responses when the OpenAI account ran out of credit. After a
two-month gap the model alias could have changed underneath, so that partial
run is archived (`results/archive_gpt54mini/`) and **not used for any result**.
The final run was restarted from zero on gpt-5.6-luna in one sitting.

## Pipeline

1. `run_local.py` — compression grid (noop, random_deletion, llmlingua_qwen,
   llmlingua2 × 0.7/0.5/0.3) over 1,456 prompts + prompt-level BERTScore.
   Deterministic and model-independent; reused from July (dataset MD5s
   verified unchanged). 57 errors, all LLMLingua-1 on oversized probe prompts
   that every analysis already excludes.
2. `features_final.py` — APCS feature vector for all 1,456 prompts (Farasa
   standalone, corrected morphology).
3. `run_api.py` — `--smoke` first (6 real prompts, cost projection), then the
   full run: seeded random call order, $6 projected-cost check after 500
   calls, $9 hard budget guard, immediate stop on insufficient quota,
   checkpoint/resume.
4. `run_chain.py` — runs, in order, stopping at the first failure:
   - `score_and_analyze.py` — output-level BERTScore (both scorers), noise
     ceiling + τ (both scorers), snapshot-drift check, RQ1 comparisons with
     Wilcoxon + paired bootstrap 95% CIs, ranking-flip replication,
     LLMLingua-2 vs LLMLingua like-for-like, ceiling-normalised fidelity,
     and an AraBERT window check: every pair where either answer exceeds
     510 wordpieces is flagged, and the main comparison is repeated without
     those pairs. (Re-check of Exp 07c, needed because the output cap rose
     from 512 to 1,024 tokens; on the archived July responses 4% of pairs
     exceed the window, mostly instruction, and excluding them moves the
     LLMLingua-2 vs random difference by ≈0.001.)
   - `rq2_final.py` — RQ2 on dev: feature correlations, and the
     length-vs-category deconfounding (within-category ρ, within-band
     Kruskal–Wallis, OLS unique-R² decomposition).
   - `apcs_final.py` — APCS recalibration on dev, then the ONE-SHOT test
     evaluation against every SDR baseline (no compression, always
     LLMLingua, always LLMLingua-2, random), with accuracy, measured USD
     cost saving, fidelity violations, SDR Pareto-hit, bootstrap CIs,
     McNemar, confusion matrix, per-category accuracy, and an mBERT
     sensitivity re-run. Ships `apcs/apcs/rules_default.json` v1.0.0-final.
   - `../09_synthetic_sensitivity/sweep_09a.py` and `analyze_09b.py`.
   - `make_figures.py` — 8 thesis figures (PNG 300 dpi + PDF) and
     `figures/tables.md`.

`stats_utils.py` holds the shared bootstrap/McNemar code.

## Verification before the paid run

The full chain was dry-run on (a) the archived partial responses — dev rows
only, test excluded — which exercised real BERTScore scoring, RQ2 and APCS
calibration on a pseudo-test carved from dev; and (b) a fabricated-outcome
fixture over all dev + probe prompts, which exercised 09a, 09b and every
figure. Dry-run outputs lived in a scratch directory and were never written
to `results/`. The dry run caught two issues fixed before launch: the
always-LLMLingua baseline could never score under the primary label set (now
compared on the extended set), and the cross-share test broke when a
threshold was identical in every repeat.

## Outputs (`results/`)

local_results.csv, features_final.csv, responses.jsonl, api_config.json,
final_results.csv, ceiling.csv, analysis_10.json, rq2_final.json,
rq2_correlations.csv, apcs_final.json, chain_log.txt, figures/, FINDINGS.md.
