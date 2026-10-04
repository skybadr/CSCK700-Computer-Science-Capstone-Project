"""Exp 11f — Secondary (reported, not pre-registered) exam analyses.

1. Protected vs plain LLMLingua-2 on exam prompts WITH a length instruction
   (the only prompts where the two differ): cost, fidelity, QA correctness.
2. QA correctness by method on the 100 exam QA prompts.
Nothing here feeds back into any selector.
Output: results/exam_secondary.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def main():
    df = pd.read_csv(RESULTS / "exam_results.csv")
    feats = pd.read_csv(RESULTS / "exam_features.csv").set_index("prompt_id")
    noop = df[df.method == "noop"].set_index("prompt_id")
    flagged = feats.index[feats.has_length_instruction == 1]
    out = {"n_with_length_instruction": int(len(flagged)), "protected_vs_plain": {},
           "qa_correct_by_method": {}}
    for r in [0.7, 0.5, 0.3]:
        rows = {}
        for m in ["llmlingua2", "llmlingua2_protected"]:
            g = df[(df.method == m) & (df.target_rate == r)
                   & df.prompt_id.isin(flagged)].set_index("prompt_id")
            base = noop.loc[g.index, "cost_usd"]
            rows[m] = {"tcr": round(float(g.tcr.mean()), 3),
                       "out_f1": round(float(g.out_f1_arabert.mean()), 3),
                       "cost_change_pct": round(100 * float(g.cost_usd.sum() / base.sum() - 1), 1)}
        out["protected_vs_plain"][f"@{r}"] = rows
    qa = df[df.category == "qa"]
    out["qa_correct_by_method"] = (qa.groupby(["method", "target_rate"]).qa_contains
                                   .mean().round(3).rename(lambda x: x).to_dict())
    out["qa_correct_by_method"] = {f"{m}@{r}": v for (m, r), v in
                                   out["qa_correct_by_method"].items()}
    (RESULTS / "exam_secondary.json").write_text(json.dumps(out, indent=2))
    print(f"exam prompts with a length instruction: {len(flagged)}")
    for k, v in out["protected_vs_plain"].items():
        p, q = v["llmlingua2"], v["llmlingua2_protected"]
        print(f"  rate {k}: cost plain {p['cost_change_pct']:+.1f}% vs protected "
              f"{q['cost_change_pct']:+.1f}% | F1 {p['out_f1']} vs {q['out_f1']} | "
              f"TCR {p['tcr']} vs {q['tcr']}")
    print("QA correct:", out["qa_correct_by_method"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
