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

## Current rules (v0.1.0-pilot)

Calibrated on the AraPromptBench 500-prompt pilot (Experiment 06):
`token_count < 120 → no compression; otherwise LLMLingua-2 @ 0.5`.
Fidelity criterion: output-level BERTScore F1 (AraBERT) ≥ 0.70 vs the
uncompressed prompt's response. Custom rules: `APCSSelector(rules_path=...)`.

## Tests

```
python -m pytest tests
```
