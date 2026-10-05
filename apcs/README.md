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

Three selectors ship with the package, all calibrated on the AraPromptBench v2
dev split (800 prompts) and frozen as data files:

| Selector | File | Needs | Optimises | Held-out result |
|---|---|---|---|---|
| `v1` APCS 1.0.0 (default) | `rules_default.json` | prompt | best balance (max TCR with output F1 ≥ τ) | 31.0% test / 31.2% exam |
| `v2` APCS-v2 | `selectors/apcs_v2.json` | prompt + task category | best balance | **38.5% exam** (beats best fixed strategy, Holm p = 0.028) |
| `cost` APCS-cost | `selectors/apcs_cost.json` | prompt + task category | cheapest call with output F1 ≥ τ | −1.0% bill, 9.2% below τ (exam) |

CLI:
```
apcs "اشرح مفهوم الذكاء الاصطناعي بأسلوب مبسط."
# recommendation : none
# rule fired     : token_count 32 < 80

apcs --selector v2 --category summarisation --file prompt.txt
# recommendation : llmlingua2 @ rate 0.5
# rule fired     : cat_creative <= 0.5 and token_count > 104.5 and cat_qa <= 0.5 and token_count <= 243

apcs --selector cost --category summarisation --file prompt.txt --json
```

Library:
```python
from apcs import APCSSelector
rec = APCSSelector("v2").recommend(prompt, category="qa")
rec.method   # "none" | "llmlingua2" | "llmlingua2_protected" | "llmlingua"
rec.rate     # target keep-rate, e.g. 0.5
rec.rule     # the guards that fired (explainability)
rec.features # extracted feature vector
APCSSelector("cost").compress(prompt, category="qa")  # needs the compression extra
```

`llmlingua2_protected` keeps every sentence containing an answer-length
instruction verbatim and compresses the rest with LLMLingua-2
(`apcs.protect.compress_protected`).

## Rules

APCS 1.0.0 (Experiment 10): `token_count < 80 → no compression; 80–249 →
LLMLingua-2 @ 0.5; ≥ 250 → LLMLingua-2 @ 0.3`. Fidelity criterion: output-level
BERTScore F1 (AraBERT) ≥ 0.65 vs the uncompressed prompt's response, τ derived
from the LLM's measured noise ceiling (gpt-5.6-luna).

APCS-v2 and APCS-cost (Experiment 11): shallow CART trees; full rules in
`selectors/*.json` (`rules_text`). Evaluated once on a fresh 400-prompt exam
set under a pre-registered protocol; `tests/test_tree_selectors.py` checks the
package reproduces the frozen exam choices exactly.

**Cost caveat:** on models that price output above input, plain LLMLingua-2
can raise the total bill because it deletes brevity instructions and answers
get longer (Experiment 10, F5). Use `cost`, or the protected compressor, when
the bill matters more than token reduction.

Custom rules: `APCSSelector("v1", rules_path=...)`.

## Example notebook

`examples/apcs_example.ipynb` walks through single and batch recommendations,
the three selectors, applying a recommendation and custom calibration.

## Overhead

Recommendation ≈0.05 ms per prompt; LLMLingua-2 compression ≈30 ms per prompt
on a consumer GPU (RTX 5070) and ≈0.4 s on CPU (Experiment 12).

## Tests

```
python -m pytest tests
```
