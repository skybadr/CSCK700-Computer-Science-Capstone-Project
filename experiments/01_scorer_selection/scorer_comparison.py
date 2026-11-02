"""Scorer-model selection experiment for AraPromptBench.

Compares two BERTScore scorer models on Arabic prompts:
  - bert-base-multilingual-cased (SDR primary)
  - aubmindlab/bert-base-arabertv02 (SDR sensitivity check)

There is no ground-truth fidelity label, so the comparison uses construct
validity: a good semantic-fidelity scorer should
  (1) decrease monotonically as content is deleted (monotonicity),
  (2) separate related-but-compressed text from unrelated text (separability),
  (3) be robust to meaning-preserving surface edits (robustness),
  (4) leave usable headroom around the dissertation's F1 >= 0.85 threshold.

Outputs: results/pair_scores.csv, results/summary.json, printed report.
"""

import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bert_score import score as bertscore
from scipy import stats

SEED = 42
SAMPLE_PER_CATEGORY = 125  # full dataset (all prompts per category)
KEEP_RATIOS = [0.9, 0.7, 0.5, 0.3]
MODELS = {
    "mbert": ("bert-base-multilingual-cased", 9),   # bert-score default layer
    "arabert": ("aubmindlab/bert-base-arabertv02", 9),  # matched layer depth
}

ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent.parent / "AraPromptBench_dataset.json"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def random_deletion(words: list[str], keep_ratio: float, rng: random.Random) -> str:
    n_keep = max(1, int(round(len(words) * keep_ratio)))
    idx = sorted(rng.sample(range(len(words)), n_keep))
    return " ".join(words[i] for i in idx)


def surface_normalise(text: str) -> str:
    """Meaning-preserving edits: strip punctuation, tatweel, extra spaces."""
    text = text.replace("ـ", "")  # tatweel
    text = re.sub(r"[،؛؟«»\"'.,:;!?()\[\]{}«»-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_pairs(prompts: list[dict]) -> pd.DataFrame:
    rng = random.Random(SEED)
    rows = []
    texts = [p["prompt"] for p in prompts]
    for i, p in enumerate(prompts):
        ref = p["prompt"]
        words = ref.split()
        rows.append((p["id"], p["category"], "identical", 1.0, ref, ref))
        for r in KEEP_RATIOS:
            rows.append((p["id"], p["category"], "deletion", r,
                         random_deletion(words, r, rng), ref))
        shuffled = words[:]
        rng.shuffle(shuffled)
        rows.append((p["id"], p["category"], "shuffled", None, " ".join(shuffled), ref))
        rows.append((p["id"], p["category"], "normalised", None, surface_normalise(ref), ref))
        # unrelated: random prompt from a different category
        others = [t for j, t in enumerate(texts) if prompts[j]["category"] != p["category"]]
        rows.append((p["id"], p["category"], "unrelated", None, rng.choice(others), ref))
    return pd.DataFrame(rows, columns=["prompt_id", "category", "perturbation",
                                       "keep_ratio", "candidate", "reference"])


def summarise(df: pd.DataFrame, model_key: str) -> dict:
    col = f"f1_{model_key}"
    g = df.groupby("perturbation")[col].mean()

    # 1. monotonicity: per-prompt Spearman between keep_ratio and F1
    #    (include identical pair as keep_ratio 1.0)
    dele = df[df["perturbation"].isin(["deletion", "identical"])]
    rhos = []
    for _, grp in dele.groupby("prompt_id"):
        if grp["keep_ratio"].nunique() > 2:
            rho, _ = stats.spearmanr(grp["keep_ratio"], grp[col])
            rhos.append(rho)
    monotonicity = float(np.mean(rhos))

    # 2. separability: deletion@0.5 (related) vs unrelated
    related = df[(df["perturbation"] == "deletion") & (df["keep_ratio"] == 0.5)][col]
    unrelated = df[df["perturbation"] == "unrelated"][col]
    pooled_sd = np.sqrt((related.var(ddof=1) + unrelated.var(ddof=1)) / 2)
    cohens_d = float((related.mean() - unrelated.mean()) / pooled_sd)
    # overlap: fraction of unrelated pairs scoring above the 5th pct of related
    auc = float(np.mean([
        (r > u) + 0.5 * (r == u)
        for r in related.to_numpy() for u in unrelated.to_numpy()
    ]))

    dynamic_range = float(g["identical"] - g["unrelated"])

    # 3. robustness: penalty for meaning-preserving edits
    robustness_penalty = float(g["identical"] - g["normalised"])

    # 4. threshold usability vs 0.85
    frac_unrelated_above_085 = float((unrelated >= 0.85).mean())
    frac_mild_above_085 = float(
        (df[(df["perturbation"] == "deletion") & (df["keep_ratio"] == 0.9)][col] >= 0.85).mean())

    return {
        "mean_f1_by_perturbation": {k: round(float(v), 4) for k, v in g.items()},
        "monotonicity_spearman": round(monotonicity, 4),
        "separability_cohens_d": round(cohens_d, 3),
        "separability_auc": round(auc, 4),
        "dynamic_range": round(dynamic_range, 4),
        "robustness_penalty": round(robustness_penalty, 4),
        "pct_unrelated_scoring_above_0.85": round(100 * frac_unrelated_above_085, 1),
        "pct_keep0.9_scoring_above_0.85": round(100 * frac_mild_above_085, 1),
    }


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    prompts = data["prompts"]
    rng = random.Random(SEED)
    sample = []
    for cat in data["metadata"]["categories"]:
        cat_prompts = [p for p in prompts if p["category"] == cat]
        sample.extend(rng.sample(cat_prompts, SAMPLE_PER_CATEGORY))
    print(f"Sampled {len(sample)} prompts ({SAMPLE_PER_CATEGORY}/category), seed {SEED}")

    df = build_pairs(sample)
    print(f"Built {len(df)} candidate/reference pairs "
          f"({sorted(df['perturbation'].unique())})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    for key, (model_name, layers) in MODELS.items():
        t0 = time.time()
        print(f"\nScoring with {model_name} (layer {layers}) ...")
        _, _, f1 = bertscore(
            df["candidate"].tolist(), df["reference"].tolist(),
            model_type=model_name, num_layers=layers,
            batch_size=64, device=device, verbose=True,
        )
        df[f"f1_{key}"] = f1.numpy()
        print(f"  done in {time.time() - t0:.1f}s")

    df.drop(columns=["candidate", "reference"]).to_csv(
        RESULTS / "pair_scores.csv", index=False, encoding="utf-8-sig")

    summary = {
        "config": {
            "seed": SEED, "n_prompts": len(sample), "keep_ratios": KEEP_RATIOS,
            "models": {k: v[0] for k, v in MODELS.items()},
            "num_layers": {k: v[1] for k, v in MODELS.items()},
            "device": device,
        },
        "results": {k: summarise(df, k) for k in MODELS},
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print(json.dumps(summary["results"], indent=2))
    print(f"\nSaved: {RESULTS / 'pair_scores.csv'}, {RESULTS / 'summary.json'}")


if __name__ == "__main__":
    main()
