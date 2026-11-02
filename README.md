# Arabic-Aware Prompt Compression Selector (APCS) and AraPromptBench

APCS is an installable Python package for selecting compression strategies
for Arabic prompts. This release also provides the datasets, experiment
scripts and stored results used to evaluate it.

Maintainer: Badr Abouabdou.

## Local workflow and release context

This is a later release and reproduction snapshot of the local project
workspace. It contains selected source files, datasets and stored outputs,
without the original development history. The snapshot itself does not
establish when the earlier evaluation protocol was fixed.

Experiment orchestration, compression, scoring and analysis ran locally.
The evaluated commercial LLM supplied answers through its remote hosted
API. Dataset construction records identify the synthetic prompt generators
and retain their provenance.

## What is here

| Path | Content |
|---|---|
| `apcs/` | **The IT artefact**: installable Python package (CLI + library), tests, user manual (`apcs/README.md`) and example notebook (`apcs/examples/apcs_example.ipynb`) |
| `demo/` | Streamlit demo interface over the package (`streamlit run demo/app.py`); replays stored exam results, no API key needed |
| `AraPromptBench_v3.json` | Combined release: 1,400 Arabic prompts, preserving 800 dev / 200 test / 400 exam |
| `AraPromptBench_v2.json` | Original 1,000-prompt benchmark used in Experiment 10 (850 from six public corpora, 150 synthetic), 800 dev / 200 test |
| `AraPromptBench_exam.json` | Fresh 400-prompt exam set evaluated under an internally pre-specified protocol |
| `AraPromptBench_dataset.json` | 500-prompt pilot set (development only) |
| `experiments/` | One folder per experiment (01–13): `README.md` (question, method), scripts, `results/` (raw outputs, configuration, `FINDINGS.md`) |
| `experiments/EXPERIMENT_LOG.md` | Index of all experiments and their outcomes |
| `experiments/AUDIT/` | Independent script that recomputes every reported number from the raw data |
| `RELEASE_INFO.json` | Release scope, execution context and provenance notes |
| `MANIFEST.sha256` | SHA-256 checksums for the release files |

## AraPromptBench v3

The combined release concatenates the original v2 and fresh-exam records.
It preserves every prompt, ID, source, licence field, gold answer and split.
The original files remain available because the experiment scripts and
reported results use their separate partitions. V3 is a later distribution
of those records; its full 1,400 prompts are not a new unseen test set.
The 500-prompt pilot and 456-prompt synthetic sensitivity pool remain separate.

Rebuild or verify the combined file from the repository root:

```
python experiments/08_dataset_construction/assemble_v3.py
python experiments/08_dataset_construction/assemble_v3.py --check
```

Source-specific terms are retained per record. No blanket licence is
assigned to the combined dataset; see
[DATA_PROVENANCE.md](DATA_PROVENANCE.md), including the unresolved source terms.

## Install the package only

From this snapshot's top-level folder (Python 3.10 or later):

```
python -m pip install ./apcs
apcs --selector v2 --category qa "<Arabic prompt>"
```

For compression as well:

```
python -m pip install "./apcs[compression]"
```

The first compression downloads the LLMLingua-2 model (about 2 GB).

## Quick start

Run these commands from the folder containing this README.

```
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
cd apcs
python -m pytest -q               # 33 tests pass; 1 optional GPU parity test is skipped
cd ..
apcs "اشرح مفهوم الذكاء الاصطناعي بأسلوب مبسط."
apcs --selector cost --category qa --file demo/example_prompt.txt
```

## Reproducing the results

Stored API responses are included in the `results/` folders, so the analyses
can be re-run locally without new inference calls:

```
python experiments/AUDIT/verify_results.py     # 119 checks against the raw data
```

To regenerate with fresh inference through the hosted API (requires an
OpenAI API key in `OPENAI_API_KEY`; the recorded final benchmark cost
US$2.49 and the exam US$0.61):
`experiments/10_final_benchmark/run_local.py`, `run_api.py`, `run_chain.py`,
then the Experiment 11 scripts in the order given in its README.

## Data licences

Source corpora keep their own terms, recorded per prompt. CIDAR declares
Apache 2.0 and also contains research-purpose wording; the sampled EASC
mirror's terms remain unresolved; ARCD declares MIT in its dataset-card
metadata but leaves its data-licensing section incomplete. Aya and TyDi QA
declare Apache 2.0; XL-Sum declares CC BY-NC-SA 4.0. Synthetic prompts:
CC BY 4.0. APCS code: MIT.

The [data provenance review](DATA_PROVENANCE.md) records the checked source
revisions and the correction of earlier CIDAR, EASC and ARCD labels. These
status labels do not establish rights to underlying source material.
