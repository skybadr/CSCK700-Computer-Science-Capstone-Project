# Experiment 07 — Protocol checks (ceiling, LLMLingua-1 probe, scorer window)

Three targeted checks closing methodological gaps flagged in Exps 02–06,
before the final ~1,000-prompt protocol is frozen.

| Part | Question | Script |
|---|---|---|
| a | What does output-level F1 score when the *same* prompt is sent twice (temp 0.0)? — the noise ceiling that any fidelity threshold must respect | `a_repeat_ceiling.py` (500 gpt-4o-mini calls, $0.087, checkpointed) |
| b | Is LLMLingua-1's short-prompt length gate bypassable via `target_token` instead of `rate`? | `b_llmlingua1_target_token.py` (20 slice prompts, local) |
| c | How much output-level scoring is affected by AraBERT's 512-wordpiece window? | `c_scorer_window.py` (local, no API) |

## Run

```
# a needs OPENAI_API_KEY in the environment
C:\Capstone Project\.venv\Scripts\python.exe a_repeat_ceiling.py
C:\Capstone Project\.venv\Scripts\python.exe b_llmlingua1_target_token.py
C:\Capstone Project\.venv\Scripts\python.exe c_scorer_window.py
```

Outputs in `results/`: `ceiling.csv` + `repeat_responses.jsonl`,
`llmlingua1_target_token.csv`, `scorer_window.csv`. Findings: `results/FINDINGS.md`.
