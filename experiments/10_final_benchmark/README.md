# Experiment 10 — Final benchmark on AraPromptBench v2 (+ probe pool)

The definitive measurement run: all headline dissertation numbers come from
here. Replaces the pilot protocol with every fix flagged in Exps 04–08.

## Protocol changes vs pilot (all documented, all evidence-driven)

| Aspect | Pilot | Final | Why |
|---|---|---|---|
| LLM under test | gpt-4o-mini | gpt-5.4-mini (auto-resolved, recorded in `results/api_config.json`) | 4o family deprecated mid-project; 5.4-mini is its tier successor |
| Output cap | 512 max_tokens | 1,024 max_completion_tokens | pilot creative truncation (Exp 04); v2 creative briefs also cap requested output ≤120 words |
| Ceiling | measured post-hoc (Exp 07) | measured in-run (repeat call per original) | τ must be derived from THIS model's noise floor |
| τ | 0.70 (validated post-hoc) | derived from ceiling at the 99.6%-specificity criterion, rounded to 0.05 | Exp 07 Finding A2 |
| Dataset | 500 pilot, unsplit | v2: 800 dev / 200 test (untouched until `apcs_final.py`) | contamination-free headline claims |
| Probe pool | — | 456 synthetic prompts measured in the SAME run | Exp 09 origin/condition deconfound |

Temperature: gpt-5.4-mini rejects explicit temperature and `reasoning_effort`
(adaptive reasoning model); determinism is therefore handled entirely by the
measured ceiling, per the Exp 07 methodology.

## Pipeline (in order)

1. `run_local.py` — compression grid (noop, random_deletion, llmlingua_qwen,
   llmlingua2 × 0.7/0.5/0.3) over 1,456 prompts + prompt-level BERTScore.
2. `run_api.py` — one response per unique prompt text + ceiling repeats;
   checkpointed (`responses.jsonl`); $60 budget guard; model auto-resolution
   with fallback chain gpt-5.4-mini → gpt-5-mini → gpt-4o-mini.
3. `score_and_analyze.py` — output-level BERTScore (AraBERT + mBERT), ceiling
   stats, τ derivation, RQ1 method comparison on v2 DEV only.
4. `apcs_final.py` — APCS recalibration on dev (Exp 06 procedure), ships
   package rules v1.0.0-final, then the ONE-SHOT held-out test evaluation
   (accuracy vs baselines, McNemar). The only script that reads test rows.

Then: `../09_synthetic_sensitivity/sweep_09a.py --results ...` and
`analyze_09b.py --results ...` (both dev+probe only).

## Outputs

`results/`: local_results.csv, responses.jsonl, api_config.json,
final_results.csv, ceiling.csv, analysis_10.json, apcs_final.json,
manifest_local.json, FINDINGS.md.
