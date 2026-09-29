"""Exp 10g — Does compression actually save money? (v2 DEV only)

Measured, not estimated: every call's real prompt and completion token usage
at the LLM-under-test's prices. Reports, per method x rate, input saving vs
total-cost change relative to no compression, the answer-length inflation
behind the gap, and a mechanism test: prompts carrying an explicit
length/brevity instruction, split by whether the compressor kept that
instruction.

Output: cost_analysis.json (in --indir).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

from stats_utils import bootstrap_ci

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
# brevity/length markers used by the v2 QA and summarisation templates and
# common in instruction prompts
LENGTH_MARKERS = re.compile(
    r"بإيجاز|موجز|جملتين|ثلاث جمل|جملة واحدة|فقرة واحدة|لا يتجاوز|كلمة")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=str(RESULTS))
    args = ap.parse_args()
    indir = Path(args.indir)
    price = json.loads((RESULTS / "api_config.json").read_text())["price_per_M"]

    df = pd.read_csv(indir / "final_results.csv")
    dev = df[(df.origin == "v2") & (df.split == "dev")].copy()
    dev["in_usd"] = dev.resp_prompt_tokens * price["in"] / 1e6
    dev["out_usd"] = dev.resp_completion_tokens * price["out"] / 1e6
    dev["total_usd"] = dev.in_usd + dev.out_usd
    noop = dev[dev.method == "noop"].set_index("prompt_id")

    out = {"price_per_M": price, "per_policy": {}, "answer_length_ratio": {},
           "length_instruction_mechanism": {}}
    base_in, base_total = noop.in_usd.sum(), noop.total_usd.sum()
    for (m, r), g in dev.groupby(["method", "target_rate"]):
        if m == "noop":
            continue
        g = g.set_index("prompt_id")
        ratio = g.resp_completion_tokens / noop.loc[g.index, "resp_completion_tokens"]
        delta = g.total_usd - noop.loc[g.index, "total_usd"]
        est, lo, hi = bootstrap_ci(delta)
        out["per_policy"][f"{m}@{r}"] = {
            "input_saving_pct": round(100 * (1 - g.in_usd.sum() / base_in), 1),
            "total_cost_change_pct": round(100 * (g.total_usd.sum() / base_total - 1), 1),
            "mean_delta_usd_per_prompt": est, "ci95": [lo, hi],
            "mean_answer_tokens": round(float(g.resp_completion_tokens.mean()), 1)}
        out["answer_length_ratio"][f"{m}@{r}"] = {
            "median": round(float(ratio.median()), 3),
            "share_longer": round(float((ratio > 1).mean()), 3),
            "median_by_category": ratio.groupby(g.category).median().round(3).to_dict()}
    out["baseline_mean_answer_tokens"] = round(float(noop.resp_completion_tokens.mean()), 1)

    has = noop[noop.compressed.str.contains(LENGTH_MARKERS)].index
    out["length_instruction_mechanism"]["n_prompts_with_instruction"] = int(len(has))
    for m in ["llmlingua2", "random_deletion"]:
        for r in [0.3, 0.5, 0.7]:
            c = dev[(dev.method == m) & (dev.target_rate == r)
                    & dev.prompt_id.isin(has)].set_index("prompt_id")
            kept = c.compressed.str.contains(LENGTH_MARKERS)
            ratio = c.resp_completion_tokens / noop.loc[c.index, "resp_completion_tokens"]
            u = stats.mannwhitneyu(ratio[~kept], ratio[kept]) \
                if kept.any() and (~kept).any() else None
            out["length_instruction_mechanism"][f"{m}@{r}"] = {
                "instruction_kept_pct": round(100 * float(kept.mean()), 1),
                "answer_ratio_kept": round(float(ratio[kept].median()), 3),
                "answer_ratio_dropped": round(float(ratio[~kept].median()), 3),
                "n_kept": int(kept.sum()), "n_dropped": int((~kept).sum()),
                "mannwhitney_p": float(u.pvalue) if u else None}

    (indir / "cost_analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, v in out["per_policy"].items():
        print(f"{k:22s} input {v['input_saving_pct']:+6.1f}% saved | total cost "
              f"{v['total_cost_change_pct']:+6.1f}% | answer x"
              f"{out['answer_length_ratio'][k]['median']}")
    for k, v in out["length_instruction_mechanism"].items():
        if isinstance(v, dict):
            print(f"{k:22s} kept {v['instruction_kept_pct']}% | answer x"
                  f"{v['answer_ratio_kept']} kept vs x{v['answer_ratio_dropped']} "
                  f"dropped | p={v['mannwhitney_p']:.2g}")
    print(f"saved cost_analysis.json to {indir}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
