# APCS demo interface

A Streamlit interface over the evaluated `apcs` package, made for demonstrating
the artefact. It follows one Arabic prompt through five steps:

| Step | What it shows |
|---|---|
| 1 · Prompt | An example from the held-out exam set (with an English gist) or your own prompt, and its task category |
| 2 · Read | The features APCS measures, the answer-length instruction highlighted in the prompt, and how a tokeniser splits Arabic |
| 3 · Decide | The recommendation of APCS 1.0.0, APCS-v2 or APCS-cost, and the decision as plain yes/no questions |
| 4 · Compress | The package compressing the prompt live: deleted words struck through, token counts before and after |
| 5 · Outcome | For exam prompts, the stored answer, billed tokens, cost and fidelity with and without APCS; the averages over all 400 exam prompts; how to call APCS from code |

It is a presentation layer only: recommendations and compression come from the
package itself (`APCSSelector.recommend` and `.compress`), and step 5 replays the
stored exam results, so it needs **no API key and makes no API calls**.

## Run

From the snapshot root, with the project virtual environment activated:

```
pip install -e "./apcs[compression]"      # compression step (LLMLingua-2; GPU optional)
pip install -r demo/requirements.txt  # Streamlit
streamlit run demo/app.py
```

The app opens at http://localhost:8501 with the light theme in
`demo/.streamlit/config.toml`. LLMLingua-2 starts loading in the background when
the app opens (about 10–20 s), so step 4 is quick by the time you reach it.

## Suggested walkthrough (6–8 minutes)

| # | Where | Point to make |
|---|---|---|
| 1 | Step 1, example **“Answer briefly”** | A real question from the held-out exam; the gist translates it |
| 2 | Step 2 | APCS reads five cheap features; «بإيجاز» (briefly) is an answer-length instruction. Open *Why Arabic needs more tokens*: وبالكتاب is one word but 6 tokens |
| 3 | Step 3, **APCS-v2** | The tree asks four questions and recommends LLMLingua-2, keeping 50% |
| 4 | Step 4, **Compress** | Deleted words are struck through, including «بإيجاز»; the output is identical to the compression evaluated in the exam |
| 5 | Step 5, open *Read both answers* | The prompt got shorter, but the answer grew from 29 to 199 tokens: the call costs 328% more |
| 6 | Back to step 3, **APCS-cost** | It protects the instruction and keeps 30%; in steps 4–5 «بإيجاز» survives and the call costs 38% less with a faithful answer |
| 7 | Step 5, table | Among the three APCS selectors, APCS-cost lowers the total bill and has the fewest answers below τ over all 400 exam prompts |
| 8 | Step 1, **Passage question** (APCS-v2) | When no length instruction is at stake, compression pays: 46% cheaper, same answer |
| 9 | Step 1, **English text**, then step 3 | Fail-safe: outside its domain it declines to compress and says why |
| 10 | Step 5, *Use APCS in your own code*; terminal: `apcs --selector cost --category qa --file demo/example_prompt.txt`, then `pytest -q` in `apcs/` (33 passed, 1 skipped) | Real, tested, installable software |

Before recording: open the app once so the model finishes loading, and zoom the
browser to 110–125%. Never show a terminal that prints environment variables.
