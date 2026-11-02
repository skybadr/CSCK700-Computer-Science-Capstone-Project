"""Exp 10e — RQ2 on the final dataset (v2 DEV only).

1. Spearman: each feature vs output-level F1, per method x rate (as Exp 05).
2. Deconfounding length from task category — the question the pilot could
   not answer because its short prompts were all qa/creative:
   a. within-category Spearman(token_count, out-F1)
   b. within-band Kruskal-Wallis across categories
   c. OLS: out-F1 ~ standardised features + C(category), with the drop-one
      change in R^2 for length vs category (unique variance explained)
3. LLMLingua-1 length gate: Spearman(token_count, achieved keep).
4. Feature inter-correlation (redundancy).

Morphological density uses the corrected definition (Exp 05 erratum).
Outputs: rq2_final.json, rq2_correlations.csv (in --indir).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FEATURES = ["token_count", "structural_complexity", "fragmentation_ratio",
            "morphological_density"]
FOCUS = ("llmlingua2", 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=str(RESULTS))
    args = ap.parse_args()
    indir = Path(args.indir)

    res = pd.read_csv(indir / "final_results.csv")
    feats = pd.read_csv(RESULTS / "features_final.csv")
    dev = res[(res.origin == "v2") & (res.split == "dev") & (res.method != "noop")]
    dev = dev.merge(feats.drop(columns=["task_category"]), on="prompt_id")
    out = {"n_dev_prompts": int(dev.prompt_id.nunique()),
           "spearman_out_f1": {}, "deconfounding": {}, "gate": {}}

    rows = []
    for (method, rate), sub in dev.groupby(["method", "target_rate"]):
        for f in FEATURES:
            rho, p = stats.spearmanr(sub[f], sub.out_f1_arabert)
            rows.append(dict(method=method, target_rate=rate, feature=f,
                             rho=round(float(rho), 3), p=float(p), n=len(sub)))
            out["spearman_out_f1"][f"{method}@{rate}/{f}"] = round(float(rho), 3)
    pd.DataFrame(rows).to_csv(indir / "rq2_correlations.csv", index=False)

    foc = dev[(dev.method == FOCUS[0]) & (dev.target_rate == FOCUS[1])].copy()
    # (a) length effect within each category
    out["deconfounding"]["within_category_rho_length"] = {
        c: {"rho": round(float(stats.spearmanr(g.token_count,
                                               g.out_f1_arabert)[0]), 3),
            "p": float(stats.spearmanr(g.token_count, g.out_f1_arabert)[1]),
            "n": len(g)}
        for c, g in foc.groupby("category")}
    # (b) category effect within each length band
    kw = {}
    for band, g in foc.groupby("band"):
        groups = [x.out_f1_arabert.values for _, x in g.groupby("category")
                  if len(x) >= 5]
        if len(groups) >= 2:
            h, p = stats.kruskal(*groups)
            kw[band] = {"H": round(float(h), 2), "p": float(p),
                        "category_means": g.groupby("category")
                        .out_f1_arabert.mean().round(4).to_dict()}
    out["deconfounding"]["within_band_kruskal_category"] = kw
    # (c) regression with unique-variance decomposition
    foc["log_tokens"] = np.log(foc.token_count)
    for f in ["log_tokens", "structural_complexity", "fragmentation_ratio",
              "morphological_density"]:
        foc[f"z_{f}"] = (foc[f] - foc[f].mean()) / foc[f].std()
    zs = "z_log_tokens + z_structural_complexity + z_fragmentation_ratio + " \
         "z_morphological_density"
    full = smf.ols(f"out_f1_arabert ~ {zs} + C(category)", data=foc).fit()
    no_len = smf.ols("out_f1_arabert ~ z_structural_complexity + "
                     "z_fragmentation_ratio + z_morphological_density + "
                     "C(category)", data=foc).fit()
    no_cat = smf.ols(f"out_f1_arabert ~ {zs}", data=foc).fit()
    ci = full.conf_int()
    out["deconfounding"]["ols_llmlingua2@0.5"] = {
        "r2_full": round(float(full.rsquared), 4),
        "unique_r2_length": round(float(full.rsquared - no_len.rsquared), 4),
        "unique_r2_category": round(float(full.rsquared - no_cat.rsquared), 4),
        "coefficients": {
            k: {"b": round(float(full.params[k]), 4),
                "ci95": [round(float(ci.loc[k, 0]), 4),
                         round(float(ci.loc[k, 1]), 4)],
                "p": float(full.pvalues[k])}
            for k in full.params.index if k != "Intercept"},
        "n": int(full.nobs)}

    q = dev[dev.method == "llmlingua_qwen"]
    for rate, g in q.groupby("target_rate"):
        rho, p = stats.spearmanr(g.token_count, g.achieved_keep)
        out["gate"][f"@{rate}"] = {"rho_tokens_vs_keep": round(float(rho), 3),
                                   "p": float(p)}
    uniq = feats[feats.prompt_id.isin(dev.prompt_id.unique())]
    out["feature_intercorrelation"] = uniq[FEATURES].corr(
        method="spearman").round(3).to_dict()

    (indir / "rq2_final.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["deconfounding"]["within_category_rho_length"], indent=1))
    print(json.dumps({k: v for k, v in out["deconfounding"]
                      ["ols_llmlingua2@0.5"].items() if k != "coefficients"},
                     indent=1))
    print(f"saved rq2_final.json to {indir}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
