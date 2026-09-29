"""Exp 09a — Synthetic-share sweep at 15% / 30% / 50%, full dev scale (800).

Tests how the synthetic-data ratio affects (1) calibrated APCS thresholds,
(2) the method ranking, and (3) outcome distributions.

Design (800-prompt mixes mirroring the dev split's category composition):
  cells: instruction 280 / qa 200 / summarisation 200 / creative 120
  share s in {15%, 30%, 50%} applies to the three MANIPULABLE categories
  (creative is structurally 100% synthetic in v2 and is held constant at its
  120 dev prompts in every mix — differences between conditions are therefore
  attributable to the manipulated share alone).
  synthetic per cell at s=50%: instruction 140, qa 100, summarisation 100.

Length-confound control: each category cell uses FIXED band quotas
(short/medium/long), identical across all share conditions. Quotas start at
the dev split's band proportions and are clamped so that 0.5*quota never
exceeds the synthetic pool for that band (deficits redistribute to bands
with spare capacity). Because quotas are constant across conditions, band
composition cannot vary with share.

B bootstrap repeats per share; each repeat: sample mix -> grid-search APCS
rule (Exp 06 template) -> record thresholds & metrics.
The v2 TEST split is never read.

Usage:
  python sweep_09a.py --dry-run                 # pool feasibility check only
  python sweep_09a.py --results <outcomes.csv>  # full sweep (after benchmark)
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
RESULTS = ROOT / "results"

TAU = 0.70          # confirmed against the new model's ceiling before use
SHARES = [0.15, 0.30, 0.50]
CELLS = {"instruction": 280, "qa": 200, "summarisation": 200}
CREATIVE_CELL = 120  # constant, all-synthetic (see module docstring)
BANDS = ["short", "medium", "long"]
CANDS = ["noop@1.0", "llmlingua2@0.7", "llmlingua2@0.5", "llmlingua2@0.3"]
T1_GRID = [40, 60, 80, 100, 120, 150]
T2_GRID = [200, 250, 300, 350, 400, 500]
RATE_GRID = [0.7, 0.5, 0.3]


def load_pools():
    """pools[cat][kind][band] -> [prompt_id]; meta[id] -> token_count."""
    v2 = json.loads((PROJECT / "AraPromptBench_v2.json").read_text(encoding="utf-8"))
    probe = json.loads((ROOT / "probe_pool.json").read_text(encoding="utf-8"))
    meta, pools = {}, {}
    for c in list(CELLS) + ["creative"]:
        pools[c] = {"sourced": {b: [] for b in BANDS},
                    "synthetic": {b: [] for b in BANDS}}
    for p in v2["prompts"]:
        meta[p["id"]] = p["token_count"]
        if p["split"] != "dev":
            continue                      # test split never enters the sweep
        kind = "synthetic" if p["source"] == "synthetic-claude" else "sourced"
        if p["length_band"] in BANDS:
            pools[p["category"]][kind][p["length_band"]].append(p["id"])
    for p in probe["prompts"]:
        meta[p["id"]] = p["token_count"]
        if p["length_band"] in BANDS:     # xlong excluded by design
            pools[p["category"]]["synthetic"][p["length_band"]].append(p["id"])
    return pools, meta


def band_quotas(pools):
    """Fixed per-band quotas per category, identical across all shares.

    Start from dev band proportions; clamp each band so the max share (50%)
    is feasible from the synthetic pool; redistribute deficits to bands with
    spare synthetic AND sourced capacity."""
    smax = max(SHARES)
    quotas = {}
    for c, cell in CELLS.items():
        dev_counts = {b: len(pools[c]["sourced"][b])
                      + len([i for i in pools[c]["synthetic"][b]
                             if i.startswith("v2-")]) for b in BANDS}
        total_dev = sum(dev_counts.values())
        q = {b: round(cell * dev_counts[b] / total_dev) for b in BANDS}
        # fix rounding drift
        q[BANDS[-1]] += cell - sum(q.values())
        # clamp + redistribute
        for _ in range(6):
            deficit = 0
            for b in BANDS:
                cap = int(len(pools[c]["synthetic"][b]) / smax)
                if q[b] > cap:
                    deficit += q[b] - cap
                    q[b] = cap
            if deficit == 0:
                break
            smin = min(SHARES)
            for b in sorted(BANDS, key=lambda b: -(
                    int(len(pools[c]["synthetic"][b]) / smax) - q[b])):
                spare_syn = int(len(pools[c]["synthetic"][b]) / smax) - q[b]
                # sourced demand peaks at the LOWEST share (1-smin sourced)
                spare_src = int(len(pools[c]["sourced"][b]) / (1 - smin)) - q[b]
                give = max(0, min(deficit, spare_syn, spare_src))
                q[b] += give
                deficit -= give
                if deficit == 0:
                    break
            if deficit > 0:
                raise RuntimeError(f"{c}: cannot place {deficit} prompts — "
                                   "pools insufficient")
        quotas[c] = q
    return quotas


def check_feasibility(pools, quotas):
    print("=== fixed band quotas (identical across all shares) ===")
    ok = True
    for c, q in quotas.items():
        print(f"{c:14s} quotas {q}  (cell {sum(q.values())}, "
              f"target {CELLS[c]})")
        for s in SHARES:
            for b in BANDS:
                need_syn = round(q[b] * s)
                need_src = q[b] - need_syn
                have_syn = len(pools[c]["synthetic"][b])
                have_src = len(pools[c]["sourced"][b])
                if need_syn > have_syn or need_src > have_src:
                    ok = False
                    print(f"  !! {c}/{b} @share {s}: need syn {need_syn} "
                          f"(have {have_syn}), src {need_src} (have {have_src})")
    n_creat = len([i for b in BANDS
                   for i in pools["creative"]["synthetic"][b]])
    print(f"creative       constant cell {CREATIVE_CELL} from {n_creat} "
          f"dev synthetic prompts")
    ok = ok and n_creat >= CREATIVE_CELL
    print("FEASIBLE" if ok else "NOT FEASIBLE")
    return ok


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
    if tok < T1:
        return "noop@1.0"
    if tok >= T2:
        return f"llmlingua2@{rl}"
    return f"llmlingua2@{rm}"


def search(ids, lab, meta):
    best, best_acc = None, -1.0
    toks = np.array([meta[p] for p in ids])
    labs = np.array([lab[p] for p in ids])
    for T1, T2, rm, rl in itertools.product(T1_GRID, T2_GRID,
                                            RATE_GRID, RATE_GRID):
        if T2 <= T1:
            continue
        pred = np.where(toks < T1, "noop@1.0",
                        np.where(toks >= T2, f"llmlingua2@{rl}",
                                 f"llmlingua2@{rm}"))
        acc = float(np.mean(pred == labs))
        if acc > best_acc:
            best_acc, best = acc, (T1, T2, rm, rl)
    return best, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", help="measured outcomes CSV (dev + probe)")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tau", type=float,
                    help="fidelity threshold; default: read analysis_10.json "
                         "next to --results")
    ap.add_argument("--outdir", default=str(RESULTS))
    args = ap.parse_args()

    global TAU
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.tau is not None:
        TAU = args.tau
    elif args.results:
        a = Path(args.results).parent / "analysis_10.json"
        if a.exists():
            TAU = json.loads(a.read_text())["tau"]["arabert"]["adopted"]
    print(f"tau = {TAU}")
    pools, meta = load_pools()
    quotas = band_quotas(pools)
    feasible = check_feasibility(pools, quotas)
    if args.dry_run:
        sys.exit(0 if feasible else 1)
    if not feasible:
        sys.exit("pools insufficient — expand probe pool first")
    if not args.results:
        sys.exit("--results required for the full sweep")

    df = pd.read_csv(args.results)
    tcr, f1 = build_tables(df)
    measured = set(tcr.index)
    for c in pools:
        for k in ("sourced", "synthetic"):
            for b in BANDS:
                pools[c][k][b] = [p for p in pools[c][k][b] if p in measured]
    quotas = band_quotas(pools)          # re-derive on measured pools
    if not check_feasibility(pools, quotas):
        sys.exit("measured pools insufficient")

    creat_pool = [i for b in BANDS for i in pools["creative"]["synthetic"][b]]
    rng = np.random.default_rng(args.seed)
    rows = []
    for share in SHARES:
        for rep in range(args.reps):
            ids = list(rng.choice(creat_pool, CREATIVE_CELL, replace=False))
            for c, q in quotas.items():
                for b in BANDS:
                    n_syn = round(q[b] * share)
                    ids += list(rng.choice(pools[c]["synthetic"][b], n_syn,
                                           replace=False))
                    ids += list(rng.choice(pools[c]["sourced"][b],
                                           q[b] - n_syn, replace=False))
            lab = labels_for(ids, tcr, f1)
            (T1, T2, rm, rl), acc = search(ids, lab, meta)
            rows.append(dict(
                share=share, rep=rep, T1=T1, T2=T2, r_mid=rm, r_long=rl,
                accuracy=round(acc, 4),
                mix_size=len(ids),
                label_noop_share=round(float(np.mean(
                    [v == "noop@1.0" for v in lab.values()])), 4),
                mean_tokens=round(float(np.mean([meta[p] for p in ids])), 1)))
        print(f"share {share:.0%}: {args.reps} reps done", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "sweep_09a_raw.csv", index=False)
    summ = out.groupby("share").agg(
        T1_median=("T1", "median"),
        T1_iqr=("T1", lambda x: x.quantile(.75) - x.quantile(.25)),
        r_mid_mode=("r_mid", lambda x: x.mode()[0]),
        r_long_mode=("r_long", lambda x: x.mode()[0]),
        acc_mean=("accuracy", "mean"),
        acc_sd=("accuracy", "std"),
        noop_label_share=("label_noop_share", "mean"),
        mean_tokens=("mean_tokens", "mean")).round(3)
    print("\n=== Threshold stability by synthetic share (800-prompt mixes) ===")
    print(summ.to_string())
    summ.to_csv(outdir / "sweep_09a_summary.csv")
    tests = {}
    for col in ["T1", "accuracy", "label_noop_share"]:
        groups = [g[col].to_numpy() for _, g in out.groupby("share")]
        if np.ptp(np.concatenate(groups)) == 0:
            tests[col] = {"note": "identical in every repeat at every share "
                                  "(perfectly stable)"}
            continue
        h, p = stats.kruskal(*groups)
        tests[col] = {"kruskal_H": round(float(h), 3), "p": float(p)}
    (outdir / "sweep_09a_tests.json").write_text(json.dumps(
        {"tau": TAU, "reps_per_share": args.reps, "tests_across_shares": tests},
        indent=2))
    print("differences across shares (Kruskal-Wallis):", tests)
    print(f"\nSaved to {outdir}: sweep_09a_raw.csv, sweep_09a_summary.csv, "
          f"sweep_09a_tests.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
