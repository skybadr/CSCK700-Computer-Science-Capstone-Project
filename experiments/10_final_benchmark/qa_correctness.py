"""Exp 10h — Task accuracy on QA prompts (proposal: "task accuracy").

Do compressed QA prompts still get CORRECT answers, not just similar ones?
Gold answers are fetched from TyDi-QA / ARCD by each v2 prompt's source id.
v2 DEV only. Per method x rate: correctness rate, share of originally
correct answers that stay correct after compression, and paired McNemar for
LLMLingua-2 vs random deletion.

Output: results/qa_correctness.json, results/qa_gold_v2.json (gold cache)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qa_metrics import contains, recall
from stats_utils import bootstrap_ci, mcnemar_exact

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
RESULTS = ROOT / "results"
PQ = "hf://datasets/{}@refs/convert/parquet/{}"


def gold_for_v2():
    cache = RESULTS / "qa_gold_v2.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    from datasets import load_dataset
    v2 = json.loads((PROJECT / "AraPromptBench_v2.json").read_text(encoding="utf-8"))
    want = {(p["source"], str(p["source_id"])): p["id"] for p in v2["prompts"]
            if p["category"] == "qa"}
    gold = {}
    for src, repo, sub, arabic_only in [
            ("TyDi-QA", "google-research-datasets/tydiqa", "secondary_task", True),
            ("ARCD", "hsseinmz/arcd", "plain_text", False)]:
        for split in ["train", "validation"]:
            d = load_dataset("parquet", data_files=PQ.format(repo, f"{sub}/{split}/*.parquet"),
                             split="train")
            for r in d:
                if arabic_only and not r["id"].startswith("arabic"):
                    continue
                key = (src, str(r["id"]))
                if key in want:
                    gold[want[key]] = list(r["answers"]["text"])
    cache.write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")
    return gold


def main():
    gold = gold_for_v2()
    df = pd.read_csv(RESULTS / "final_results.csv")
    dev = df[(df.origin == "v2") & (df.split == "dev") & (df.category == "qa")].copy()
    dev = dev[dev.prompt_id.isin(gold)]
    dev["contains"] = [contains(a, gold[p]) for a, p in zip(dev.response, dev.prompt_id)]
    dev["recall"] = [recall(a, gold[p]) for a, p in zip(dev.response, dev.prompt_id)]
    base = dev[dev.method == "noop"].set_index("prompt_id")
    out = {"n_prompts": int(base.shape[0]),
           "gold_found": len(gold),
           "baseline_contains": round(float(base.contains.mean()), 4),
           "baseline_recall": round(float(base.recall.mean()), 4),
           "per_policy": {}, "llmlingua2_vs_random": {}}
    for (m, r), g in dev.groupby(["method", "target_rate"]):
        if m == "noop":
            continue
        g = g.set_index("prompt_id")
        was_ok = base.loc[g.index, "contains"] == 1
        est, lo, hi = bootstrap_ci(g.contains)
        out["per_policy"][f"{m}@{r}"] = {
            "contains": round(est, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "recall": round(float(g.recall.mean()), 4),
            "kept_correct_pct": round(100 * float(g.contains[was_ok].mean()), 1)}
    for r in [0.3, 0.5, 0.7]:
        a = dev[(dev.method == "llmlingua2") & (dev.target_rate == r)].set_index("prompt_id")
        b = dev[(dev.method == "random_deletion") & (dev.target_rate == r)].set_index("prompt_id")
        common = a.index.intersection(b.index)
        ao, bo, p = mcnemar_exact(a.loc[common, "contains"] == 1,
                                  b.loc[common, "contains"] == 1)
        out["llmlingua2_vs_random"][f"@{r}"] = {
            "llmlingua2": round(float(a.loc[common, "contains"].mean()), 4),
            "random": round(float(b.loc[common, "contains"].mean()), 4),
            "llmlingua2_only_correct": ao, "random_only_correct": bo,
            "mcnemar_p": p}
    (RESULTS / "qa_correctness.json").write_text(json.dumps(out, indent=2))
    print(f"uncompressed: answer contains gold {out['baseline_contains']:.1%}, "
          f"gold-word recall {out['baseline_recall']:.1%} (n={out['n_prompts']})")
    for k, v in out["per_policy"].items():
        print(f"  {k:22s} contains {v['contains']:.1%} {v['ci95']} | recall "
              f"{v['recall']:.1%} | still correct {v['kept_correct_pct']}%")
    for k, v in out["llmlingua2_vs_random"].items():
        print(f"  LL2 vs random {k}: {v['llmlingua2']:.1%} vs {v['random']:.1%}, "
              f"p={v['mcnemar_p']:.2g}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
