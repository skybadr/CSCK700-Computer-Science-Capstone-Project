# Experiment 05 — Feature extraction & RQ2 analysis

**Question (RQ2):** How do prompt characteristics (length, structure,
morphology, tokeniser fragmentation) affect compression efficiency and output
fidelity? Also produces the per-prompt **Pareto labels** (ground truth for
APCS rule derivation, RQ3).

## Parts

1. `compute_features.py` — the six prompt features for all 500 prompts →
   `results/features.csv`. Morphological density uses **Farasa in standalone
   (temp-file) mode**: interactive mode corrupts Arabic through the Windows
   process pipe (mojibake / stdin.flush OSError) — documented integration
   finding. Prompts are flattened to single lines and batched 100/call with
   line-alignment verification. Requires Java; the original environment used Microsoft OpenJDK 21.
   The Farasa jar (~241 MB) is cached on first use.
2. `analyze.py` — merges features with Exp 03 (TCR) and Exp 04 (output-level
   F1); Spearman correlations per method × rate (`results/correlations.csv`);
   per-prompt Pareto frontier over (TCR, out-F1-AraBERT) including the noop
   anchor (`results/pareto_labels.csv`).

The original morphology calculation counted punctuation and numeral
tokens. Erratum 05e in `results/FINDINGS.md` and `recheck_density.py` record
the correction; final feature analysis in Experiment 10 uses the corrected
measure. The scripts below retain the original pilot workflow.

## Run

From the snapshot root, with the project virtual environment activated:

```
# java must be on PATH (e.g. C:\Program Files\Microsoft\jdk-21*\bin)
python experiments/05_feature_analysis/compute_features.py
python experiments/05_feature_analysis/analyze.py
```

Findings and the draft APCS rule sketch: `results/FINDINGS.md`.
