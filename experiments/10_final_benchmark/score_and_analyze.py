"""Exp 10c — Merge responses, output-level scoring, ceiling, tau, RQ1 stats.

Produces:
  results/final_results.csv   all rows (v2 + probe) with out-F1 both scorers
  results/ceiling.csv         repeat-call ceiling per prompt
  results/analysis_10.json    ceiling stats, tau decision, method comparison
                              (v2 DEV only — test rows are scored but excluded
                              from every aggregate here; they are consumed
                              once, by apcs_final.py)
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TAU_SPEC = 0.996          # specificity criterion from Exp 07 (pilot tau=0.70)


def key_of(pid, text, repeat=False):
    h = hashlib.md5(f"{pid}|{text}".encode("utf-8")).hexdigest()
    return f"{h}-r2" if repeat else h


def main():
    df = pd.read_csv(RESULTS / "local_results.csv")
    df = df[(df.error.fillna("") == "") & (df.compressed.fillna("") != "")].copy()
    recs = [json.loads(l) for l in
            (RESULTS / "responses.jsonl").open(encoding="utf-8") if l.strip()]
    resp = {r["key"]: r["response"] for r in recs}
    trunc = sum(1 for r in recs if r["finish_reason"] == "length")
    in_tok = sum(r["prompt_tokens"] for r in recs)
    out_tok = sum(r["completion_tokens"] for r in recs)

    df["call_key"] = [key_of(r.prompt_id, r.compressed) for r in df.itertuples()]
    df["response"] = df.call_key.map(resp)
    missing = int(df.response.isna().sum())
    if missing:
        print(f"WARNING: {missing} rows without responses — dropped")
        df = df[df.response.notna()]

    noop = df[df.method == "noop"]
    ref_resp = noop.set_index("prompt_id").response
    df["ref_response"] = df.prompt_id.map(ref_resp)
    # ceiling second responses
    ceil_rows = []
    for r in noop.itertuples():
        k2 = key_of(r.prompt_id, r.compressed, repeat=True)
        if k2 in resp:
            ceil_rows.append(dict(prompt_id=r.prompt_id, category=r.category,
                                  origin=r.origin, split=r.split,
                                  r1=r.response, r2=resp[k2]))
    ceil = pd.DataFrame(ceil_rows)
    print(f"rows {len(df)} | ceiling pairs {len(ceil)} | truncated {trunc}")

    from bert_score import score as bertscore
    scored = df.method != "noop"
    for key, model in [("arabert", "aubmindlab/bert-base-arabertv02"),
                       ("mbert", "bert-base-multilingual-cased")]:
        print(f"output-level BERTScore ({key}) on {int(scored.sum())} rows ...",
              flush=True)
        _, _, f1 = bertscore(df.loc[scored, "response"].tolist(),
                             df.loc[scored, "ref_response"].tolist(),
                             model_type=model, num_layers=9, batch_size=32,
                             device=DEVICE)
        df.loc[scored, f"out_f1_{key}"] = f1.numpy().round(4)
        df.loc[~scored, f"out_f1_{key}"] = 1.0
    print("ceiling BERTScore (arabert) ...", flush=True)
    _, _, f1 = bertscore(ceil.r2.tolist(), ceil.r1.tolist(),
                         model_type="aubmindlab/bert-base-arabertv02",
                         num_layers=9, batch_size=32, device=DEVICE)
    ceil["ceiling_f1_arabert"] = f1.numpy().round(4)

    df.drop(columns=["ref_response"]).to_csv(
        RESULTS / "final_results.csv", index=False, encoding="utf-8-sig")
    ceil.drop(columns=["r1", "r2"]).to_csv(
        RESULTS / "ceiling.csv", index=False, encoding="utf-8-sig")

    # ---------------- analysis (v2 DEV rows only) ---------------------------
    dev = df[(df.origin == "v2") & (df.split == "dev") & scored]
    cdev = ceil[(ceil.origin == "v2") & (ceil.split == "dev")]
    s = cdev.ceiling_f1_arabert
    tau_exact = float(np.quantile(s, 1 - TAU_SPEC))
    tau = round(round(tau_exact / 0.05) * 0.05, 2)
    analysis = {
        "api_usage": {"responses": len(recs), "prompt_tokens": in_tok,
                      "completion_tokens": out_tok, "truncated": trunc},
        "ceiling_dev": {
            "n": len(cdev), "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "p25": round(float(s.quantile(.25)), 4),
            "p5": round(float(s.quantile(.05)), 4),
            "identical_rate_note": "string-identity not stored; see medians",
            "by_category_median": {k: round(float(v), 4) for k, v in
                                   cdev.groupby("category")
                                   .ceiling_f1_arabert.median().items()},
        },
        "tau": {"specificity_criterion": TAU_SPEC,
                "tau_exact_quantile": round(tau_exact, 4),
                "tau_adopted": tau,
                "pilot_tau": 0.70},
        "method_comparison_dev": {},
    }
    for rate in [0.3, 0.5, 0.7]:
        a = dev[(dev.method == "llmlingua2") & (dev.target_rate == rate)] \
            .set_index("prompt_id").out_f1_arabert
        b = dev[(dev.method == "random_deletion") & (dev.target_rate == rate)] \
            .set_index("prompt_id").out_f1_arabert.reindex(a.index).dropna()
        a = a.reindex(b.index)
        w = stats.wilcoxon(a, b)
        analysis["method_comparison_dev"][f"llmlingua2_vs_random@{rate}"] = {
            "llmlingua2_mean": round(float(a.mean()), 4),
            "random_mean": round(float(b.mean()), 4),
            "diff": round(float(a.mean() - b.mean()), 4),
            "wilcoxon_p": float(w.pvalue), "n": int(len(a))}
    g = dev.groupby(["method", "target_rate"])[
        ["achieved_keep", "tcr", "out_f1_arabert", "out_f1_mbert"]].mean().round(4)
    analysis["dev_means"] = {f"{m}@{r}": v for (m, r), v in
                             g.to_dict("index").items()}
    (RESULTS / "analysis_10.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False))
    print(json.dumps(analysis["ceiling_dev"], indent=2))
    print(json.dumps(analysis["tau"], indent=2))
    for k, v in analysis["method_comparison_dev"].items():
        print(k, v)
    print("saved: final_results.csv, ceiling.csv, analysis_10.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
