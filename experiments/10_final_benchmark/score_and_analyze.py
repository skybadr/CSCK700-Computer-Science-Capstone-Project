"""Exp 10c — Merge responses, output-level scoring, noise ceiling, tau, RQ1.

Produces (in --outdir, default results/):
  final_results.csv   every row (v2 + probe) with out-F1 under both scorers
                      and the measured API usage of its response
  ceiling.csv         repeat-call ceiling per prompt, both scorers
  analysis_10.json    ceiling stats, tau (both scorers), snapshot drift check,
                      RQ1 method comparisons with Wilcoxon + bootstrap CIs

Every aggregate here uses v2 DEV rows only. Test rows are scored (so the
one-shot evaluation can read them) but never summarised in this script.

--dry-run: pipeline check on the archived partial gpt-5.4-mini responses;
test rows are dropped entirely, a missing ceiling falls back to tau = 0.70,
and nothing is written to results/.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from stats_utils import paired_bootstrap_ci

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TAU_SPEC = 0.996          # Exp 07: tau fails at most 0.4% of pure-noise pairs
WINDOW = 510              # AraBERT max wordpieces excluding [CLS]/[SEP]
SCORERS = {"arabert": "aubmindlab/bert-base-arabertv02",
           "mbert": "bert-base-multilingual-cased"}


def key_of(pid, text, repeat=False):
    h = hashlib.md5(f"{pid}|{text}".encode("utf-8")).hexdigest()
    return f"{h}-r2" if repeat else h


def bertscore_f1(cands, refs, scorer):
    from bert_score import score as bertscore
    _, _, f1 = bertscore(cands, refs, model_type=SCORERS[scorer], num_layers=9,
                         batch_size=32, device=DEVICE)
    return f1.numpy().round(4)


def tau_from_ceiling(s):
    exact = float(np.quantile(s, 1 - TAU_SPEC))
    return exact, round(round(exact / 0.05) * 0.05, 2)


def compare(dev, a_sel, b_sel, col):
    """Paired comparison on the prompts both selections cover."""
    a = dev.loc[a_sel].set_index("prompt_id")[col]
    b = dev.loc[b_sel].set_index("prompt_id")[col]
    common = a.index.intersection(b.index)
    if len(common) < 10:
        return {"n": int(len(common)), "note": "too few paired prompts"}
    a, b = a.loc[common], b.loc[common]
    est, lo, hi = paired_bootstrap_ci(a, b)
    return {"n": int(len(common)), "a_mean": round(float(a.mean()), 4),
            "b_mean": round(float(b.mean()), 4),
            "diff": round(est, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default=str(RESULTS / "responses.jsonl"))
    ap.add_argument("--outdir", default=str(RESULTS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS / "local_results.csv")
    df = df[(df.error.fillna("") == "") & (df.compressed.fillna("") != "")].copy()
    if args.dry_run:
        df = df[df.split != "test"]

    recs = [json.loads(l) for l in open(args.responses, encoding="utf-8")
            if l.strip()]
    by_key = {r["key"]: r for r in recs}
    snapshots = sorted({r.get("model", "unrecorded") for r in recs})

    df["call_key"] = [key_of(r.prompt_id, r.compressed) for r in df.itertuples()]
    df["response"] = df.call_key.map(lambda k: by_key.get(k, {}).get("response"))
    df["resp_prompt_tokens"] = df.call_key.map(
        lambda k: by_key.get(k, {}).get("prompt_tokens"))
    df["resp_completion_tokens"] = df.call_key.map(
        lambda k: by_key.get(k, {}).get("completion_tokens"))
    df["resp_truncated"] = df.call_key.map(
        lambda k: by_key.get(k, {}).get("finish_reason") == "length")

    # a prompt is usable only if its uncompressed reference was answered
    ref = df[df.method == "noop"].set_index("prompt_id").response.dropna()
    n_before = df.prompt_id.nunique()
    df = df[df.prompt_id.isin(ref.index) & df.response.notna()].copy()
    df["ref_response"] = df.prompt_id.map(ref)
    print(f"rows {len(df)} | prompts {df.prompt_id.nunique()}/{n_before} usable "
          f"| snapshots {snapshots}")

    # AraBERT reads at most 510 wordpieces; longer answers are silently cut
    # before scoring. Flag every pair where either side exceeds the window.
    from transformers import AutoTokenizer
    wp_tok = AutoTokenizer.from_pretrained(SCORERS["arabert"])
    uniq = pd.unique(pd.concat([df.response, df.ref_response]))
    wp = {t: len(wp_tok(t, add_special_tokens=False)["input_ids"]) for t in uniq}
    df["window_exceeded"] = ((df.response.map(wp) > WINDOW)
                             | (df.ref_response.map(wp) > WINDOW))

    scored = df.method != "noop"
    for s in SCORERS:
        print(f"output-level BERTScore ({s}) on {int(scored.sum())} rows ...",
              flush=True)
        df.loc[scored, f"out_f1_{s}"] = bertscore_f1(
            df.loc[scored, "response"].tolist(),
            df.loc[scored, "ref_response"].tolist(), s)
        df.loc[~scored, f"out_f1_{s}"] = 1.0

    # repeat-call ceiling
    noop = df[df.method == "noop"]
    ceil = pd.DataFrame([
        dict(prompt_id=r.prompt_id, category=r.category, origin=r.origin,
             split=r.split, r1=r.response,
             r2=by_key[key_of(r.prompt_id, r.compressed, True)]["response"])
        for r in noop.itertuples()
        if key_of(r.prompt_id, r.compressed, True) in by_key])
    tau = {}
    if len(ceil):
        for s in SCORERS:
            ceil[f"ceiling_f1_{s}"] = bertscore_f1(ceil.r2.tolist(),
                                                   ceil.r1.tolist(), s)
        ceil["identical"] = ceil.r1 == ceil.r2
        cdev = ceil[(ceil.origin == "v2") & (ceil.split == "dev")]
        for s in SCORERS:
            exact, adopted = tau_from_ceiling(cdev[f"ceiling_f1_{s}"])
            tau[s] = {"exact_quantile": round(exact, 4), "adopted": adopted}
    elif args.dry_run:
        cdev = pd.DataFrame()
        tau = {s: {"exact_quantile": None, "adopted": 0.70,
                   "note": "dry-run fallback, no ceiling calls"} for s in SCORERS}
    else:
        sys.exit("no ceiling responses found — the run is incomplete")

    df.drop(columns=["ref_response"]).to_csv(
        outdir / "final_results.csv", index=False, encoding="utf-8-sig")
    if len(ceil):
        ceil.drop(columns=["r1", "r2"]).to_csv(
            outdir / "ceiling.csv", index=False, encoding="utf-8-sig")

    # ------------------------------ RQ1 (dev only) ---------------------------
    dev = df[(df.origin == "v2") & (df.split == "dev")]
    m, r = dev.method, dev.target_rate
    analysis = {
        "api": {"responses": len(recs), "snapshots": snapshots,
                "snapshot_drift": len(snapshots) > 1,
                "prompt_tokens": int(sum(x["prompt_tokens"] for x in recs)),
                "completion_tokens": int(sum(x["completion_tokens"]
                                             for x in recs)),
                "truncated": int(sum(x["finish_reason"] == "length"
                                     for x in recs))},
        "tau": tau, "tau_spec": TAU_SPEC, "pilot_tau": 0.70,
        "arabert_window": {
            "wordpiece_limit": WINDOW,
            "dev_scored_pairs_exceeding_pct": round(100 * float(
                dev[dev.method != "noop"].window_exceeded.mean()), 2),
            "by_category_pct": (100 * dev[dev.method != "noop"]
                                .groupby("category").window_exceeded.mean())
            .round(2).to_dict()},
        "rq1_output_level": {}, "rq1_prompt_level": {},
        "llmlingua2_vs_llmlingua1_like_for_like": {},
    }
    if len(cdev):
        analysis["ceiling_dev"] = {
            s: {"median": round(float(cdev[f"ceiling_f1_{s}"].median()), 4),
                "mean": round(float(cdev[f"ceiling_f1_{s}"].mean()), 4),
                "p5": round(float(cdev[f"ceiling_f1_{s}"].quantile(.05)), 4),
                "pct_below_0.85": round(100 * float(
                    (cdev[f"ceiling_f1_{s}"] < 0.85).mean()), 1)}
            for s in SCORERS}
        analysis["ceiling_dev"]["identical_response_pct"] = round(
            100 * float(cdev.identical.mean()), 1)
        cat_ceil = cdev.groupby("category").ceiling_f1_arabert.median()
        ll = dev[(m == "llmlingua2") & (r == 0.5)].groupby("category") \
            .out_f1_arabert.mean()
        analysis["ceiling_normalised_llmlingua2@0.5"] = {
            c: {"ceiling_median": round(float(cat_ceil[c]), 4),
                "out_f1": round(float(ll[c]), 4),
                "retained_pct": round(100 * float(ll[c] / cat_ceil[c]), 1)}
            for c in cat_ceil.index if c in ll.index}

    for rate in [0.3, 0.5, 0.7]:
        a_sel = (m == "llmlingua2") & (r == rate)
        b_sel = (m == "random_deletion") & (r == rate)
        analysis["rq1_output_level"][f"llmlingua2_vs_random@{rate}"] = \
            compare(dev, a_sel, b_sel, "out_f1_arabert")
        analysis["rq1_output_level"][f"llmlingua2_vs_random@{rate}_mbert"] = \
            compare(dev, a_sel, b_sel, "out_f1_mbert")
        analysis["rq1_prompt_level"][f"llmlingua2_vs_random@{rate}"] = \
            compare(dev, a_sel, b_sel, "f1_arabert")
        ok = ~dev.window_exceeded
        analysis["rq1_output_level"][
            f"llmlingua2_vs_random@{rate}_within_window"] = \
            compare(dev, a_sel & ok, b_sel & ok, "out_f1_arabert")
        # like-for-like: only prompts where LLMLingua-1 actually compressed
        compressed = dev[(m == "llmlingua_qwen") & (r == rate)
                         & (dev.achieved_keep < 0.95)].prompt_id
        q_sel = (m == "llmlingua_qwen") & (r == rate) & dev.prompt_id.isin(compressed)
        l_sel = (m == "llmlingua2") & (r == rate) & dev.prompt_id.isin(compressed)
        res = compare(dev, l_sel, q_sel, "out_f1_arabert")
        res["keep_llmlingua2"] = round(float(dev.loc[l_sel].achieved_keep.mean()), 3)
        res["keep_llmlingua1"] = round(float(dev.loc[q_sel].achieved_keep.mean()), 3)
        analysis["llmlingua2_vs_llmlingua1_like_for_like"][f"@{rate}"] = res

    g = dev.groupby(["method", "target_rate"])[
        ["achieved_keep", "tcr", "f1_arabert", "out_f1_arabert",
         "out_f1_mbert"]].mean().round(4)
    analysis["dev_means"] = {f"{k[0]}@{k[1]}": v
                             for k, v in g.to_dict("index").items()}
    gate = dev[m == "llmlingua_qwen"].groupby("category").apply(
        lambda x: round(100 * float((x.achieved_keep > 0.95).mean()), 1),
        include_groups=False)
    analysis["llmlingua1_uncompressed_pct_by_category"] = gate.to_dict()

    (outdir / "analysis_10.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: analysis[k] for k in
                      ["api", "tau", "rq1_output_level"]}, indent=1)[:3000])
    print(f"saved to {outdir}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
