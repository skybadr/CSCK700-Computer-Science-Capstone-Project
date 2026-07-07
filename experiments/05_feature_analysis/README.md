# Experiment 05 — Feature extraction & RQ2 analysis

**Question (RQ2):** How do prompt characteristics (length, structure,
morphology, tokeniser fragmentation) affect compression efficiency and output
fidelity? Also produces the per-prompt **Pareto labels** (ground truth for
APCS rule derivation, RQ3).

## Parts

1. `compute_features.py` — the six SDR features for all 500 prompts →
   `results/features.csv`. Morphological density uses **Farasa in standalone
   (temp-file) mode**: interactive mode corrupts Arabic through the Windows
   process pipe (mojibake / stdin.flush OSError) — documented integration
   finding. Prompts are flattened to single lines and batched 100/call with
   line-alignment verification. Requires Java (Microsoft OpenJDK 21 installed
   via winget; Farasa jar ~241 MB cached on first run).
2. `analyze.py` — merges features with Exp 03 (TCR) and Exp 04 (output-level
   F1); Spearman correlations per method × rate (`results/correlations.csv`);
   per-prompt Pareto frontier over (TCR, out-F1-AraBERT) including the noop
   anchor, per SDR Algorithm 2 (`results/pareto_labels.csv`).

## Run

```
# java must be on PATH (e.g. C:\Program Files\Microsoft\jdk-21*\bin)
C:\Capstone Project\.venv\Scripts\python.exe compute_features.py
C:\Capstone Project\.venv\Scripts\python.exe analyze.py
```

Findings and the draft APCS rule sketch: `results/FINDINGS.md`.
