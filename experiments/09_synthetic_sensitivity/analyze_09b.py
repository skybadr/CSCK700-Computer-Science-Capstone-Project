"""Exp 09b — Synthetic-vs-sourced comparison, deconfounded.

Compares compression outcomes on synthetic probe prompts vs corpus-sourced v2
DEV prompts *within* category (instruction/qa/summarisation) and length band.
Creative is excluded (no sourced counterpart exists). Test split never read.

Per category x rate (llmlingua2, the APCS-relevant method):
  - Mann-Whitney U on out_f1_arabert, synthetic vs sourced (in-band rows)
  - band-stratified means + achieved-keep comparison (rate-control parity)
  - Spearman of token_count vs out_f1 within each origin (slope similarity)

Usage: python analyze_09b.py --results <final_results.csv>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CATS = ["instruction", "qa", "summarisation"]
BANDS = ["short", "medium", "long"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", default=str(RESULTS))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results)
    df = df[df.method == "llmlingua2"]
    src = df[(df.origin == "v2") & (df.split == "dev")
             & df.category.isin(CATS)]
    syn = df[(df.origin == "probe") & df.category.isin(CATS)
             & df.band.isin(BANDS)]
    print(f"sourced rows {len(src)} | synthetic rows {len(syn)}")

    out = {"per_category_rate": {}, "band_means": {}, "length_slopes": {},
           "rate_control": {}}
    for c in CATS:
        for rate in [0.3, 0.5, 0.7]:
            a = syn[(syn.category == c) & (syn.target_rate == rate)].out_f1_arabert
            b = src[(src.category == c) & (src.target_rate == rate)].out_f1_arabert
            u = stats.mannwhitneyu(a, b, alternative="two-sided")
            adj, lo, hi = band_adjusted_diff(
                syn[(syn.category == c) & (syn.target_rate == rate)],
                src[(src.category == c) & (src.target_rate == rate)])
            out["per_category_rate"][f"{c}@{rate}"] = {
                "synthetic_mean": round(float(a.mean()), 4), "n_syn": len(a),
                "sourced_mean": round(float(b.mean()), 4), "n_src": len(b),
                "diff_raw": round(float(a.mean() - b.mean()), 4),
                "diff_band_adjusted": round(adj, 4),
                "ci95_band_adjusted": [round(lo, 4), round(hi, 4)],
                "rank_biserial": round(1 - 2 * float(u.statistic)
                                       / (len(a) * len(b)), 3),
                "mannwhitney_p": float(u.pvalue)}
        for b_ in BANDS:
            aa = syn[(syn.category == c) & (syn.band == b_)].out_f1_arabert
            bb = src[(src.category == c) & (src.band == b_)].out_f1_arabert
            out["band_means"][f"{c}/{b_}"] = {
                "synthetic": round(float(aa.mean()), 4) if len(aa) else None,
                "sourced": round(float(bb.mean()), 4) if len(bb) else None}
        for name, d in [("synthetic", syn), ("sourced", src)]:
            sub = d[d.category == c]
            rho, p = stats.spearmanr(sub.orig_tokens, sub.out_f1_arabert)
            out["length_slopes"][f"{c}/{name}"] = {
                "spearman_rho": round(float(rho), 3), "p": float(p)}
        ka = syn[syn.category == c].groupby("target_rate").achieved_keep.mean()
        kb = src[src.category == c].groupby("target_rate").achieved_keep.mean()
        out["rate_control"][c] = {
            str(r): {"synthetic": round(float(ka.get(r, np.nan)), 3),
                     "sourced": round(float(kb.get(r, np.nan)), 3)}
            for r in [0.3, 0.5, 0.7]}

    # multiple-comparison note: 9 primary tests -> Bonferroni alpha 0.0056
    sig = {k: v for k, v in out["per_category_rate"].items()
           if v["mannwhitney_p"] < 0.05 / 9}
    out["bonferroni_significant"] = list(sig)
    out["band_adjusted_ci_excludes_zero"] = [
        k for k, v in out["per_category_rate"].items()
        if v["ci95_band_adjusted"][0] > 0 or v["ci95_band_adjusted"][1] < 0]
    (outdir / "analysis_09b.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out["per_category_rate"], indent=2))
    print("bonferroni-significant (raw):", list(sig))
    print("band-adjusted CI excludes 0:", out["band_adjusted_ci_excludes_zero"])
    print("saved:", outdir / "analysis_09b.json")


def band_adjusted_diff(syn, src, n_boot=2000, seed=42):
    """Synthetic minus sourced mean out-F1, weighting each length band by the
    SOURCED band mix, so a different length distribution cannot masquerade
    as an origin effect. Bootstrap resamples within band. Bands with no
    synthetic rows are dropped (and their weight renormalised)."""
    bands = [b for b in BANDS
             if (syn.band == b).any() and (src.band == b).any()]
    w = np.array([(src.band == b).sum() for b in bands], float)
    w /= w.sum()
    sa = [syn[syn.band == b].out_f1_arabert.to_numpy() for b in bands]
    sb = [src[src.band == b].out_f1_arabert.to_numpy() for b in bands]
    est = float(sum(wi * (a.mean() - b.mean()) for wi, a, b in zip(w, sa, sb)))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        boots.append(sum(
            wi * (rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean())
            for wi, a, b in zip(w, sa, sb)))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return est, float(lo), float(hi)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
