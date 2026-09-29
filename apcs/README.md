# APCS — Arabic-Aware Prompt Compression Selector

Given an Arabic prompt, APCS recommends the prompt-compression strategy
(method + target keep-rate) that best balances token savings against semantic
fidelity, using rules calibrated empirically on AraPromptBench.

## Install

```
pip install -e .            # core (tiktoken only)
pip install -e .[compression]  # + llmlingua, enables APCSSelector.compress()
pip install -e .[morphology]   # + farasapy (needs Java) for morphological density
```

## Usage

CLI:
```
apcs "اشرح مفهوم الذكاء الاصطناعي بأسلوب مبسط."
# recommendation : none            (32 tokens — too short to compress safely)

apcs --file long_prompt.txt --json
```

Library:
```python
from apcs import APCSSelector
rec = APCSSelector().recommend(prompt)
rec.method   # "none" | "llmlingua2"
rec.rate     # target keep-rate, e.g. 0.5
rec.rule     # the guard that fired (explainability)
rec.features # extracted feature vector
```

## Current rules (v1.0.0-final)

Calibrated on the AraPromptBench v2 dev split (800 prompts) with
gpt-5.6-luna as the LLM under test (Experiment 10):
`token_count < 80 → no compression; 80–249 → LLMLingua-2 @ 0.5;
≥ 250 → LLMLingua-2 @ 0.3`. Fidelity criterion: output-level BERTScore F1
(AraBERT) ≥ 0.65 vs the uncompressed prompt's response, τ derived from the
model's measured noise ceiling. Held-out test accuracy 31.0% (95% CI
24.5–37.5%); see `experiments/10_final_benchmark/results/FINDINGS.md`.

**Cost caveat:** the rule optimises token reduction, not the bill. On models
that price output well above input, LLMLingua-2 can raise total cost because
it tends to delete brevity instructions and answers get longer (Exp 10, F5).

The pilot rule (v0.1.0) is preserved in git history. Custom rules:
`APCSSelector(rules_path=...)`.

## Tests

```
python -m pytest tests
```
