# Setup Guide

This is a step-by-step setup guide assuming you have nothing installed.

## 1. Install Python

You need Python 3.10 or newer.

- **Windows / macOS:** download from [python.org](https://www.python.org/downloads/)
- **Linux:** use your package manager (`apt`, `dnf`, etc.)

Verify:
```bash
python --version    # should print 3.10 or higher
```

## 2. Install Git

- **Windows:** [git-scm.com](https://git-scm.com/download/win)
- **macOS:** `brew install git` or install Xcode Command Line Tools
- **Linux:** `sudo apt install git`

## 3. Clone or initialise the repo

If you're starting fresh:
```bash
cd /path/to/your/projects
# Move this folder here and rename if you like
cd arapromptbench
git init
git add .
git commit -m "Initial scaffold"
```

When you're ready, push to GitHub:
```bash
gh repo create arapromptbench --private --source=. --push
# or manually create on github.com and `git remote add origin ...`
```

## 4. Create a virtual environment

```bash
python -m venv .venv

# Activate it:
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows (PowerShell or cmd)
```

You should see `(.venv)` at the start of your terminal prompt.

## 5. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will take a few minutes and install ~3–4 GB of libraries (PyTorch, transformers, etc.).

> **Note on PyTorch:** if you have a GPU and want CUDA support, install PyTorch separately first following [pytorch.org/get-started](https://pytorch.org/get-started/locally/), *then* run `pip install -r requirements.txt`.

## 6. Get API access

You need at least one LLM API key. Recommendations:

- **OpenAI** — `gpt-4o-mini` is cheap (~$0.15 / 1M input tokens) and Arabic-capable. Sign up at [platform.openai.com](https://platform.openai.com/).
- **Anthropic** — `claude-haiku-4-5` is comparably priced and strong on Arabic. Sign up at [console.anthropic.com](https://console.anthropic.com/).

For your dissertation budget, set a usage cap:
- OpenAI: Settings → Limits → set monthly spend limit (e.g., $20)
- Anthropic: Workspace → Spend limits

## 7. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and paste your key(s):

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

**Never commit `.env` to git.** It's in `.gitignore` already.

## 8. Run the vertical slice

```bash
python -m experiments.001_vertical_slice.run
```

What it does:
1. Loads ~20 seed Arabic prompts from `data/prompts/seed_prompts.json`
2. Compresses each with the chosen method(s)
3. Sends both original and compressed versions to the LLM
4. Computes TCR, BERTScore, and output similarity
5. Writes results to `results/001_vertical_slice/results.csv`

You can also run it in **dry mode** (no API calls, just compression and metrics):
```bash
python -m experiments.001_vertical_slice.run --dry-run
```

This is useful before you have API access set up, or to validate the pipeline.

## Common issues

**`ModuleNotFoundError: No module named 'src'`**
Run from the repo root, not from inside a subfolder.

**`bert-score` downloads a huge model on first use**
Yes — first call downloads `roberta-large` (~1.4 GB). Subsequent calls use the cache.

**LLMLingua is slow on CPU**
It loads a small LM internally for token scoring. CPU is fine for ~100 prompts; for the full 1200, consider Colab or a small GPU.

**Arabic text shows as `?????` in CSV**
Open the CSV with UTF-8 encoding. In Excel: Data → From Text → choose UTF-8. In pandas: `pd.read_csv("results.csv", encoding="utf-8")`.
