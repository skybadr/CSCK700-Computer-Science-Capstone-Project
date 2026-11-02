# Experiment 11 — Pre-registration (fresh exam)

Committed to git **before** any exam prompt is sent to the LLM. Nothing below
may change after the exam run; deviations, if any, are reported as such.

## Data

`AraPromptBench_exam.json`: 400 never-seen prompts (140 instruction / 100
summarisation / 100 QA / 60 creative), built like AraPromptBench v2, with
zero exact, partial (12-word) or record overlap with the pilot, v2 or the
probe pool (`results/overlap_audit.json`). Not used for any design decision.

## Protocol

LLM gpt-5.6-luna, reasoning_effort none, temperature 0, 1,024 output tokens
(identical to Exp 10, imported from `run_api.py`). Fidelity = output-level
BERTScore-F1 (AraBERT, layer 9) vs the answer to the uncompressed prompt.
**τ = 0.65**, fixed from Exp 10's dev noise ceiling. Cost = measured tokens
× $0.20 / $1.20 per M (input / output). Methods per prompt: no compression;
random deletion, LLMLingua (Qwen), LLMLingua-2 and LLMLingua-2-protected at
0.7 / 0.5 / 0.3. Script: `run_exam.py`; evaluation: `evaluate_exam.py`.

## Frozen selectors (`results/selectors/`)

| File | Role | Definition |
|---|---|---|
| `apcs_v1.json` | **Replication** | APCS 1.0.0 as shipped: tokens < 80 → none; 80–249 → LLMLingua-2 @ 0.5; ≥ 250 → @ 0.3 |
| `apcs_v2.json` | **Primary** | depth-4 decision tree (min leaf 20) over token count, structure, fragmentation, morphology, has-length-instruction and task category; trained on v2 dev with best-balance labels; dev 10-fold CV accuracy 43.5% |
| `apcs_cost.json` | **Extension** | depth-3 tree (min leaf 40), same features, trained on cheapest-faithful labels over 10 candidates incl. the protected variant; dev CV cost −1.9% vs none |

Labels on the exam are computed exactly as in design:
best balance = largest TCR among {none, LLMLingua-2 @ 0.7/0.5/0.3} with
F1 ≥ τ, else none; cheapest faithful = lowest measured cost among the 10
candidates with F1 ≥ τ (none always eligible; ties → larger TCR).

## Hypotheses (Holm correction across H1–H3, α = 0.05)

- **H1 (primary):** APCS-v2 best-balance accuracy > always LLMLingua-2 @ 0.5
  (the best fixed strategy on dev). Exact McNemar; supported if Holm-adjusted
  p < 0.05 and APCS-v2 ahead.
- **H2:** APCS-v2 accuracy > APCS-1.0.0. Exact McNemar; same criterion.
- **H3 (extension):** APCS-cost lowers the measured total bill vs never
  compressing. One-sided Wilcoxon signed-rank on per-prompt cost
  differences; supported if Holm-adjusted p < 0.05 and total cost change < 0.

## Reported regardless of outcome

- **R1 replication:** APCS-1.0.0 exam accuracy with 95% CI, compared with its
  31.0% on the v2 test split.
- Every policy (3 selectors, 10 fixed strategies, random): best-balance
  accuracy with 95% CI, cheapest-faithful accuracy, mean TCR, mean output F1,
  share below τ, total cost change vs none, QA correctness (answer contains
  gold) on the 100 QA prompts.
- APCS-v2 vs every other policy (paired McNemar + bootstrap CI), as
  secondary, uncorrected, clearly labelled.
- Accuracy by category for APCS-1.0.0, APCS-v2 and the H1 comparator.

No selector, threshold, label definition or test is altered after the exam
answers are collected. The exam set is used once.
