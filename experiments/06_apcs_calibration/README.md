# Experiment 06 — APCS rule calibration & baseline comparison (RQ3, pilot)

**Question (RQ3):** Can a feature-based rule accurately select the best
compression strategy for a given Arabic prompt?

## Method

Every (prompt, method, rate) outcome was measured in Exps 03/04, so policies
are evaluated **by lookup** on the 500-prompt pilot (in-sample; final
evaluation happens on the held-out split of the ~1,000-prompt dataset).

- **Candidates:** noop, llmlingua2 @ {0.7, 0.5, 0.3}. LLMLingua-1 excluded
  (never uniquely optimal like-for-like, Exp 04 F2); random deletion excluded
  (dominated, Exp 04 F1).
- **Ground-truth label ("best balance", per proposal):** max TCR among
  candidates with output-F1-AraBERT ≥ τ; noop if none. τ* = 0.70 (working
  value, provisional pending the repeat-call ceiling; sensitivity at 0.65/0.75
  reported). The SDR's raw Pareto-hit metric is reported alongside but is
  degenerate: noop is always on the frontier, so "never compress" scores ~100%.
- **Rule template (from Exp 05):** thresholds on token_count; grid search over
  T1 ∈ {40…150}, T2 ∈ {200…500}, rates ∈ {0.3, 0.5, 0.7}; per-category variant
  also searched.
- **Statistics:** McNemar exact test (binomial on discordant pairs) for
  accuracy vs baselines.

## Run

```
C:\Capstone Project\.venv\Scripts\python.exe calibrate.py
```

Outputs `results/calibration.json`, `results/policy_comparison.csv`, and ships
`apcs/apcs/rules_default.json` to the package. Findings: `results/FINDINGS.md`.
