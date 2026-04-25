# Experiment 001 — Vertical Slice

**Purpose:** Validate the end-to-end pipeline before scaling up.

## What this run does

- Loads ~20 Arabic prompts from `data/prompts/seed_prompts.json`
- Compresses each prompt with three methods:
  - `no_op` (baseline, no compression)
  - `random_deletion` (baseline, random tokens removed at target ratio)
  - `llmlingua` (LLMLingua with TinyLlama scorer)
- Sends both original and compressed prompts to the configured LLM
- Computes:
  - **TCR** (Token Compression Ratio)
  - **Prompt-level BERTScore** (similarity between original and compressed prompts)
  - **Output-level BERTScore** (similarity between LLM outputs)

## How to run

```bash
# Full run (requires API key)
python -m experiments.001_vertical_slice.run

# Dry run — no LLM calls (good for first time)
python -m experiments.001_vertical_slice.run --dry-run

# Baselines only (skip LLMLingua — fastest first run)
python -m experiments.001_vertical_slice.run --dry-run --no-llmlingua

# Custom compression target
python -m experiments.001_vertical_slice.run --target-ratio 0.3
```

## What to look for in the output

`results/001_vertical_slice/results.csv` should contain one row per (prompt × method).

**Sanity checks:**
- `no_op` should have `tcr ≈ 0` and `prompt_bertscore_f1 ≈ 1.0`
- `random_deletion` should have `tcr ≈ 1 - target_ratio` and lower `prompt_bertscore_f1`
- `llmlingua` should have similar `tcr` to random but **higher** `prompt_bertscore_f1`
- If LLMLingua doesn't beat random on output-level F1, that's already a finding worth investigating

## Cost estimate

- Per prompt: 2 LLM calls (original + compressed)
- Per run: 20 prompts × 3 methods × 2 calls = 120 calls
- With `gpt-4o-mini`: typically under $0.10 per full run

## Notes for future iterations

- This experiment is intentionally minimal. Don't add features here — copy to `002_*` when scaling up.
- Once stable, freeze this experiment and treat it as a regression test.
