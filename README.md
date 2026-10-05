# Arabic-Aware Prompt Compression Selector (APCS) and AraPromptBench

MSc Artificial Intelligence dissertation project (University of Liverpool,
CSCK700), Badr Abouabdou: *Design and Evaluation of an Arabic-Aware Prompt
Compression Selector for Large Language Models*.

## What is here

| Path | Content |
|---|---|
| `apcs/` | **The IT artefact**: installable Python package (CLI + library), tests, user manual (`apcs/README.md`) and example notebook (`apcs/examples/apcs_example.ipynb`) |
| `AraPromptBench_v2.json` | Final benchmark: 1,000 Arabic prompts (850 from six public corpora, 150 synthetic), 800 dev / 200 test |
| `AraPromptBench_exam.json` | Fresh 400-prompt exam set used for the pre-registered evaluation |
| `AraPromptBench_dataset.json` | 500-prompt pilot set (development only) |
| `experiments/` | One folder per experiment (01–12): `README.md` (question, method), scripts, `results/` (raw outputs, configuration, `FINDINGS.md`) |
| `experiments/EXPERIMENT_LOG.md` | Index of all experiments and their outcomes |
| `experiments/AUDIT/` | Independent script that recomputes every reported number from the raw data |
| `dissertation/` | Dissertation sources (Markdown), figures and build scripts |

## Quick start

```
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
cd apcs && python -m pytest -q    # 18 tests
apcs "اشرح مفهوم الذكاء الاصطناعي بأسلوب مبسط."
apcs --selector v2 --category summarisation --file prompt.txt
```

## Reproducing the results

All API responses are stored in the `results/` folders, so every analysis can
be re-run offline:

```
python experiments/AUDIT/verify_results.py     # 117 checks against the raw data
```

To regenerate from scratch (requires an OpenAI API key in `OPENAI_API_KEY`;
the final benchmark cost US$2.49 and the exam US$0.61):
`experiments/10_final_benchmark/run_local.py`, `run_api.py`, `run_chain.py`,
then the Experiment 11 scripts in the order given in its README.

## Data licences

Source corpora keep their own licences (CIDAR CC BY-NC 4.0; Aya Apache 2.0;
XL-Sum CC BY-NC-SA 4.0; EASC research use; TyDi QA Apache 2.0; ARCD CC BY-SA
4.0), recorded per prompt. Synthetic prompts: CC BY 4.0. APCS code: MIT.
