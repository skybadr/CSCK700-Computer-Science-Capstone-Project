# Experiment 06 — APCS rule calibration & baseline comparison (RQ3, pilot)

**Question (RQ3):** Can a feature-based rule accurately select the best
compression strategy for a given Arabic prompt?

## Method

Every (prompt, method, rate) outcome was measured in Exps 03/04, so policies
are evaluated **by lookup** on the 500-prompt pilot (in-sample; final
evaluation was subsequently completed in Experiments 10 and 11 on the
1,000-prompt benchmark and a separate 400-prompt exam).

- **Candidates:** noop, llmlingua2 @ {0.7, 0.5, 0.3}. LLMLingua-1 excluded
  (never uniquely optimal like-for-like, Exp 04 F2); random deletion excluded
  (dominated, Exp 04 F1).
- **Ground-truth label ("best balance"):** max TCR among
  candidates with output-F1-AraBERT ≥ τ; noop if none. τ* = 0.70 (working
  value at this pilot stage; Experiment 07 subsequently checked the repeat-call
  ceiling, and sensitivity at 0.65/0.75 is reported). The original raw
  Pareto-hit metric is reported alongside but is
  degenerate: noop is always on the frontier, so "never compress" scores ~100%.
- **Rule template (from Exp 05):** thresholds on token_count; grid search over
  T1 ∈ {40…150}, T2 ∈ {200…500}, rates ∈ {0.3, 0.5, 0.7}; per-category variant
  also searched.
- **Statistics:** McNemar exact test (binomial on discordant pairs) for
  accuracy vs baselines.

## Run

From the snapshot root, with the project virtual environment activated:

```
python experiments/06_apcs_calibration/calibrate.py
```

Outputs `results/calibration.json` and `results/policy_comparison.csv`.
The script also writes its pilot rule to `apcs/apcs/rules_default.json`;
replay this pilot in a separate working copy to preserve the evaluated final
rule supplied with this snapshot. That final rule comes from Experiment 10.
Findings: `results/FINDINGS.md`.
