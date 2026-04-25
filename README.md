# AraPromptBench & APCS

**Arabic-Aware Prompt Compression Selector**
MSc Artificial Intelligence — Capstone Project
University of Liverpool

This repository contains:

1. **AraPromptBench** — a benchmark dataset of Arabic prompts categorised by task type, used to evaluate prompt compression techniques.
2. **APCS (Arabic-Aware Prompt Compression Selector)** — a Python tool that recommends the most suitable compression strategy for a given Arabic prompt based on its features.

---

## Project structure

```
arapromptbench/
├── data/
│   ├── prompts/            # Curated Arabic prompts (AraPromptBench)
│   ├── raw/                # Raw collected data (gitignored, large)
│   └── processed/          # Cleaned, tokenised, feature-extracted prompts
├── src/
│   ├── compression/        # Compression method wrappers (LLMLingua, baseline, etc.)
│   ├── llm/                # LLM API clients (OpenAI, Anthropic, etc.)
│   ├── evaluation/         # Metrics: TCR, BERTScore, feature extraction
│   ├── pipeline/           # End-to-end experiment runner
│   └── utils/              # I/O, logging, helpers
├── experiments/
│   └── 001_vertical_slice/ # First end-to-end run (~20 prompts, 1 method)
├── results/                # Experiment outputs (CSV, JSON, plots)
├── docs/                   # Design notes, methodology drafts
└── tests/                  # Unit tests
```

## Quick start

See [`SETUP.md`](./SETUP.md) for full installation instructions.

```bash
# 1. Create environment
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate            # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (or ANTHROPIC_API_KEY)

# 4. Run the vertical-slice experiment
python -m experiments.001_vertical_slice.run
```

Output will be written to `results/001_vertical_slice/results.csv`.

## Roadmap

| Phase | Status |
|---|---|
| Vertical slice (20 prompts, 1 method, full pipeline) | In progress |
| Scale to 100 prompts, add LongLLMLingua + baseline | Pending |
| Full AraPromptBench (~800–1200 prompts) | Pending |
| APCS feature extraction + decision logic | Pending |
| APCS evaluation on held-out test set | Pending |
| Dissertation writing | In parallel |

## Citation

If this work supports your research, please cite the dissertation (forthcoming, 2026).
