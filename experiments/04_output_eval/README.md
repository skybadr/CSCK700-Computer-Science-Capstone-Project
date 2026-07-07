# Experiment 04 — Output-level evaluation (gpt-4o-mini)

**Question (RQ1, decisive half):** When gpt-4o-mini answers a *compressed*
Arabic prompt, how similar is its response to the response it gives the
*original* prompt? This output-level BERTScore F1 is the fidelity axis for
method ranking and Pareto labelling — Experiments 02/03 showed prompt-level F1
is biased toward verbatim retention and cannot rank methods.

## Protocol (per SDR evaluation protocol, Table 3)

- LLM under test: gpt-4o-mini, temperature 0.0, max_tokens 512, single user
  message containing the (original or compressed) prompt, no system prompt.
- One API call per unique (prompt_id, text): 500 originals + distinct
  compressed variants from Experiment 03 (texts LLMLingua-1 returned unchanged
  reuse the original's response — same input, temperature 0).
- Output similarity: BERTScore F1 between response-to-compressed and
  response-to-original; AraBERT-v02 primary, mBERT sensitivity (Experiment 01).
- Checkpointing: every response appended to `responses.jsonl`; rerunning the
  script resumes without re-calling (no double spend).
- Actual token usage and cost logged to `api_usage.json`.

Pre-run estimate from real token counts: 4,173 calls, ~553k input tokens,
**≤ $1.36** (worst case, all responses at 512 output tokens).

## Run

```
$env:OPENAI_API_KEY = [Environment]::GetEnvironmentVariable('OPENAI_API_KEY','User')
C:\Capstone Project\.venv\Scripts\python.exe run_output_eval.py
```
