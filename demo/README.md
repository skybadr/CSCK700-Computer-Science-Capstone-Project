# APCS demo interface

A Streamlit interface over the evaluated `apcs` package, made for demonstrating
the artefact. It is a presentation layer only: recommendations come from the
same frozen selectors (`apcs/apcs/selectors/`), and the evidence tab replays the
stored exam results, so it needs **no API key and makes no API calls**.

## Run

From the repository root, in the project virtual environment:

```
pip install -e apcs[compression]      # compression step (LLMLingua-2; GPU optional)
pip install -r demo/requirements.txt  # Streamlit
streamlit run demo/app.py
```

The app opens at http://localhost:8501. The first **Compress** click loads the
LLMLingua-2 model (about 10–20 s); later clicks are immediate.

## Tabs

1. **Why Arabic costs more:** token-by-token view of an Arabic word and its
   English equivalent under cl100k_base and o200k_base.
2. **Recommend & compress:** paste a prompt (or pick an example), choose a
   selector and task category, and see the recommendation, the rule that fired
   and the features; apply it and compare original and compressed text, with
   length-instruction markers highlighted.
3. **Evidence from the exam:** total cost change of plain vs length-protected
   LLMLingua-2 (reproduces Table 18), and a prompt-by-prompt view of the stored
   answers, billed tokens, cost and fidelity (exam-qa-043 reproduces Table 22).
4. **Batch:** upload a CSV (`prompt`, `category`) and download recommendations.

## Suggested walkthrough (6–8 minutes)

| # | Click | Point to make |
|---|---|---|
| 1 | Tab 1, default word وبالكتاب | One Arabic word carries a whole English phrase; cl100k_base splits it into 6 pieces |
| 2 | Terminal: `pip install -e apcs`, `pytest -q` (24 passed), `apcs "…"` | Real, tested, installable software |
| 3 | Tab 2 → example "Short instruction" | Short prompts are left alone; the rule shows why |
| 4 | Example "Summary with a length instruction", selector APCS-v2, then APCS-cost | Two objectives: v2 picks plain @ 0.5, cost picks protected @ 0.3; open "Compare all three selectors" |
| 5 | **Compress** (with APCS-cost), then the "Same rate with LLMLingua-2" expander | Protected keeps فقرة واحدة ("one paragraph"); plain may delete it |
| 6 | Tab 3, chart | Plain compression raises the bill on these prompts; protection turns it into a saving |
| 7 | Tab 3, exam-qa-043 at rate 0.5, open "Model's answer" | "Briefly" deleted → answer 23 → 85 tokens, call +193%; protected −27% |
| 8 | Tab 2 → "English text" and "Encoding damage" | Fail-safe: outside its domain it declines to compress and says why |
| 9 | Tab 4 → download the sample CSV, upload it | How a team would run it over many prompts |
| 10 | Sidebar note | Limits: route factual QA to no compression; recalibrate when model, prices or τ change |

Before recording: start the app, click **Compress** once so the model is loaded,
and zoom the browser to 125–150%. Never show a terminal that prints environment
variables.
