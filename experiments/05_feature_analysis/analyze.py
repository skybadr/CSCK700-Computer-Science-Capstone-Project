"""Experiment 05b — RQ2 analysis: features vs compression outcomes + Pareto labels.

Inputs: results/features.csv (05a), Exp03 benchmark_results.csv (TCR),
Exp04 output_eval_results.csv (output-level F1).

Produces:
  results/pareto_labels.csv   per (prompt, method, rate): outcomes + is_pareto flag
  results/correlations.csv    Spearman rho of each feature vs out-F1 / achieved keep
  printed summary tables for FINDINGS.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT_EVAL = ROOT.parent / "04_output_eval" / "results" / "output_eval_results.csv"

FEATURES = ["character_length", "token_count", "fragmentation_ratio",
            "structural_complexity", "morphological_density"]


def pareto_flags(group: pd.DataFrame) -> pd.Series:
    """Non-dominated (tcr, out_f1) pairs within one prompt's candidate set."""
    pts = group[["tcr", "out_f1_arabert"]].to_numpy()
    flags = []
    for i, (t, f) in enumerate(pts):
        dominated = np.any((pts[:, 0] >= t) & (pts[:, 1] >= f)
                           & ((pts[:, 0] > t) | (pts[:, 1] > f)))
        flags.append(not dominated)
    return pd.Series(flags, index=group.index)


def main() -> None:
    feats = pd.read_csv(RESULTS / "features.csv")
    df = pd.read_csv(OUT_EVAL)
    df = df.merge(feats.drop(columns=["subcategory"]), on="prompt_id")

    # ---------- Pareto labelling (SDR Algorithm 2, on output-level F1) ------
    cand = df.copy()  # noop included: (tcr 0, f1 1.0) anchor
    cand["is_pareto"] = cand.groupby("prompt_id", group_keys=False).apply(
        pareto_flags, include_groups=False)
    keep_cols = ["prompt_id", "task_category", "method", "target_rate",
                 "achieved_keep", "tcr", "f1_arabert", "out_f1_arabert",
                 "out_f1_mbert", "is_pareto"] + FEATURES
    cand[keep_cols].to_csv(RESULTS / "pareto_labels.csv", index=False,
                           encoding="utf-8-sig")

    nn = cand[cand.method != "noop"]
    print("=== Pareto-optimal rate by method (% of prompts where (method,rate) "
          "is on the frontier) ===")
    tab = (100 * nn.groupby(["method", "target_rate"]).is_pareto.mean()
           ).round(1).unstack()
    print(tab.to_string())

    print("\n=== Non-noop Pareto membership by category "
          "(mean # non-noop frontier points per prompt) ===")
    per_prompt = nn[nn.is_pareto].groupby(
        ["prompt_id", "task_category"]).size().rename("n_pareto").reset_index()
    base = feats[["prompt_id", "task_category"]].merge(
        per_prompt, how="left", on=["prompt_id", "task_category"]).fillna(0)
    print(base.groupby("task_category").n_pareto.mean().round(2).to_string())

    # which method dominates the frontier per category
    print("\n=== Frontier composition by category (% of frontier points) ===")
    fc = nn[nn.is_pareto].groupby(["task_category", "method"]).size()
    fc = (100 * fc / fc.groupby(level=0).sum()).round(1).unstack().fillna(0)
    print(fc.to_string())

    # ---------- Feature correlations (RQ2) ---------------------------------
    print("\n=== Spearman rho: feature vs output-level F1 (AraBERT), by method ===")
    rows = []
    for method in ["llmlingua2", "llmlingua_qwen", "random_deletion"]:
        for rate in [0.3, 0.5, 0.7]:
            sub = df[(df.method == method) & (df.target_rate == rate)]
            for f in FEATURES:
                rho, pval = stats.spearmanr(sub[f], sub.out_f1_arabert)
                rows.append(dict(method=method, target_rate=rate, feature=f,
                                 outcome="out_f1_arabert",
                                 rho=round(rho, 3), p=pval))
    # LLMLingua-1 length gate: feature vs achieved_keep
    for rate in [0.3, 0.5, 0.7]:
        sub = df[(df.method == "llmlingua_qwen") & (df.target_rate == rate)]
        for f in FEATURES:
            rho, pval = stats.spearmanr(sub[f], sub.achieved_keep)
            rows.append(dict(method="llmlingua_qwen", target_rate=rate,
                             feature=f, outcome="achieved_keep",
                             rho=round(rho, 3), p=pval))
    cor = pd.DataFrame(rows)
    cor.to_csv(RESULTS / "correlations.csv", index=False)

    piv = cor[cor.outcome == "out_f1_arabert"].pivot_table(
        index="feature", columns=["method", "target_rate"], values="rho")
    print(piv.round(2).to_string())

    print("\n=== Spearman rho: feature vs llmlingua_qwen achieved_keep "
          "(length-gate check) ===")
    piv2 = cor[cor.outcome == "achieved_keep"].pivot_table(
        index="feature", columns="target_rate", values="rho")
    print(piv2.round(2).to_string())

    # feature inter-correlation (redundancy check)
    print("\n=== Feature inter-correlation (Spearman, 500 prompts) ===")
    print(feats[FEATURES].corr(method="spearman").round(2).to_string())

    print(f"\nSaved: {RESULTS/'pareto_labels.csv'}, {RESULTS/'correlations.csv'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
