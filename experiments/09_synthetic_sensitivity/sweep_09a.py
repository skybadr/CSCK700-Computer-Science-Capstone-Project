"""Exp 09a — Synthetic-share sweep: 15% / 30% / 50%.

Tests how the synthetic-data ratio affects (1) calibrated APCS thresholds,
(2) the method ranking, and (3) outcome distributions — holding total size
and category balance constant.

Design (capped by probe availability: instruction has 30 synthetic prompts):
  mix size 240 = 60 per category (instruction/qa/summarisation/creative)
  share s in {15%, 30%, 50%}  ->  9 / 18 / 30 synthetic per category cell
  sourced remainder drawn from v2 DEV split only; synthetic drawn from the
  probe pool (+ v2 dev creative for the creative cell)
  B bootstrap repeats per share; each repeat: sample mix -> grid-search APCS
  rule (same template/grids as Exp 06) -> record thresholds & metrics.

The v2 TEST split is never read by this script.

Usage (after the main benchmark run produces measured outcomes):
  python sweep_09a.py --results <benchmark_output_eval.csv> [--reps 200]

The results CSV must contain rows for v2 dev + probe prompts with columns:
  prompt_id, method, target_rate, tcr, out_f1_arabert
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
RESULTS = ROOT / "results"

TAU = 0.70          # to be confirmed against the new model's measured ceiling
CELL = 60           # prompts per category per mix
SHARES = {0.15: 9, 0.30: 18, 0.50: 30}
CANDS = ["noop@1.0", "llmlingua2@0.7", "llmlingua2@0.5", "llmlingua2@0.3"]
T1_GRID = [40, 60, 80, 100, 120, 150]
T2_GRID = [200, 250, 300, 350, 400, 500]
RATE_GRID = [0.7, 0.5, 0.3]
CATS = ["instruction", "qa", "summarisation", "creative"]


def load_pools():
    v2 = json.loads((PROJECT / "AraPromptBench_v2.json").read_text(encoding="utf-8"))
    probe = json.loads((ROOT / "probe_pool.json").read_text(encoding="utf-8"))
    meta = {}
    pools = {c: {"sourced": [], "synthetic": []} for c in CATS}
    for p in v2["prompts"]:
        meta[p["id"]] = {"category": p["category"],
                         "token_count": p["token_count"]}
        if p["split"] != "dev":
            continue                     # test split never enters the sweep
        kind = "synthetic" if p["source"] == "synthetic-claude" else "sourced"
        pools[p["category"]][kind].append(p["id"])
    for p in probe["prompts"]:
        meta[p["id"]] = {"category": p["category"],
                         "token_count": p["token_count"]}
        if p["length_band"] == "xlong":
            continue                     # no sourced counterparts >650 tokens
        pools[p["category"]]["synthetic"].append(p["id"])
    return pools, meta


def build_tables(df):
    df = df.copy()
    df["cand"] = df.method + "@" + df.target_rate.astype(str)
    sub = df[df.cand.isin(CANDS)]
    return (sub.pivot_table(index="prompt_id", columns="cand", values="tcr"),
            sub.pivot_table(index="prompt_id", columns="cand",
                            values="out_f1_arabert"))


def labels_for(ids, tcr, f1):
    lab = {}
    for p in ids:
        ok = [c for c in CANDS if f1.loc[p, c] >= TAU]
        lab[p] = max(ok, key=lambda c: tcr.loc[p, c]) if ok else "noop@1.0"
    return lab


def rule(tok, T1, T2, rm, rl):
    if tok < T1: return "noop@1.0"
    if tok >= T2: return f"llmlingua2@{rl}"
    return f"llmlingua2@{rm}"


def search(ids, lab, meta):
    best, best_acc = None, -1.0
    toks = {p: meta[p]["token_count"] for p in ids}
    for T1, T2, rm, rl in itertools.product(T1_GRID, T2_GRID,
                                            RATE_GRID, RATE_GRID):
        if T2 <= T1:
            continue
        acc = np.mean([rule(toks[p], T1, T2, rm, rl) == lab[p] for p in ids])
        if acc > best_acc:
            best_acc, best = acc, (T1, T2, rm, rl)
    return best, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="measured outcomes CSV (v2 dev + probe rows)")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    pools, meta = load_pools()
    df = pd.read_csv(args.results)
    tcr, f1 = build_tables(df)
    measured = set(tcr.index)

    for c in CATS:
        for k in ("sourced", "synthetic"):
            pools[c][k] = [p for p in pools[c][k] if p in measured]
        print(f"{c}: sourced {len(pools[c]['sourced'])}, "
              f"synthetic {len(pools[c]['synthetic'])}")
        need = max(SHARES.values())
        assert len(pools[c]["synthetic"]) >= need, \
            f"{c}: need {need} measured synthetic prompts"
        assert len(pools[c]["sourced"]) >= CELL - min(SHARES.values())

    rng = np.random.default_rng(args.seed)
    rows = []
    # method-ranking margin on the full pools, per origin (share-independent
    # descriptive): mean out-F1 gap llmlingua2@0.5 minus random handled in 09b;
    # here we sweep calibration.
    for share, n_syn in SHARES.items():
        for b in range(args.reps):
            ids = []
            for c in CATS:
                ids += list(rng.choice(pools[c]["synthetic"], n_syn,
                                       replace=False))
                ids += list(rng.choice(pools[c]["sourced"], CELL - n_syn,
                                       replace=False))
            lab = labels_for(ids, tcr, f1)
            (T1, T2, rm, rl), acc = search(ids, lab, meta)
            noop_share = np.mean([v == "noop@1.0" for v in lab.values()])
            rows.append(dict(share=share, rep=b, T1=T1, T2=T2, r_mid=rm,
                             r_long=rl, accuracy=round(float(acc), 4),
                             label_noop_share=round(float(noop_share), 4)))
        print(f"share {share:.0%}: {args.reps} reps done", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "sweep_09a_raw.csv", index=False)

    print("\n=== Threshold stability by synthetic share ===")
    summ = out.groupby("share").agg(
        T1_median=("T1", "median"), T1_iqr=("T1", lambda x: x.quantile(.75) - x.quantile(.25)),
        r_mid_mode=("r_mid", lambda x: x.mode()[0]),
        r_long_mode=("r_long", lambda x: x.mode()[0]),
        acc_mean=("accuracy", "mean"),
        noop_label_share=("label_noop_share", "mean")).round(3)
    print(summ.to_string())
    summ.to_csv(RESULTS / "sweep_09a_summary.csv")
    print(f"\nSaved: {RESULTS/'sweep_09a_raw.csv'}, sweep_09a_summary.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
