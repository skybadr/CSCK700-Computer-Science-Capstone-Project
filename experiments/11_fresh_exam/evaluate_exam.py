"""Exp 11e — Pre-registered evaluation of the fresh exam (see PREREGISTRATION.md).

Reads frozen selectors (results/selectors/*.json) and exam_results.csv, and
evaluates every policy once. Hypothesis tests, comparators and the Holm
correction are fixed here and in PREREGISTRATION.md before the exam run.

--dev-check: code check only. Runs the same evaluation on the v2 DEV data
(where the selectors were trained, so its numbers are meaningless as
results) and writes to a scratch directory.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
EXP10 = ROOT.parent / "10_final_benchmark" / "results"
RESULTS = ROOT / "results"
SEL = RESULTS / "selectors"
sys.path.insert(0, str(ROOT.parent / "10_final_benchmark"))
from stats_utils import bootstrap_ci, mcnemar_exact, paired_bootstrap_ci  # noqa: E402

TAU = json.loads((EXP10 / "analysis_10.json").read_text())["tau"]["arabert"]["adopted"]
PRIMARY = ["noop@1.0", "llmlingua2@0.7", "llmlingua2@0.5", "llmlingua2@0.3"]
PROTECTED = [f"llmlingua2_protected@{r}" for r in (0.7, 0.5, 0.3)]
LLMLINGUA = [f"llmlingua_qwen@{r}" for r in (0.7, 0.5, 0.3)]
COST = PRIMARY + PROTECTED + LLMLINGUA
H1_COMPARATOR = "always llmlingua2@0.5"   # best fixed strategy on dev (34.5%)
SEED = 42


def walk(tree, row):
    while "leaf" not in tree:
        tree = tree["le"] if row[tree["feature"]] <= tree["threshold"] else tree["gt"]
    return tree["leaf"]


def apply_v1(rules, tok):
    t = rules["thresholds"]
    return np.where(tok < t["T1"], "noop@1.0", np.where(
        tok >= t["T2"], f"llmlingua2@{t['r_long']}", f"llmlingua2@{t['r_mid']}"))


def tables(df):
    df = df.copy()
    df["cand"] = df.method + "@" + df.target_rate.astype(str)
    return {v: df.pivot_table(index="prompt_id", columns="cand", values=v)
            for v in ["tcr", "out_f1_arabert", "cost_usd", "qa_contains"]}


def labels(piv, ids, kind):
    tcr, f1, cost = piv["tcr"], piv["out_f1_arabert"], piv["cost_usd"]
    out = []
    for p in ids:
        if kind == "best_balance":
            ok = [c for c in PRIMARY if f1.loc[p, c] >= TAU]
            out.append(max(ok, key=lambda c: tcr.loc[p, c]) if ok else "noop@1.0")
        else:
            ok = [c for c in COST if c in f1.columns and pd.notna(f1.loc[p, c])
                  and f1.loc[p, c] >= TAU] or ["noop@1.0"]
            out.append(min(ok, key=lambda c: (cost.loc[p, c], -tcr.loc[p, c])))
    return pd.Series(out, index=ids)


def evaluate(piv, feats, cats, outdir):
    ids = [p for p in feats.index if p in piv["tcr"].index
           and piv["out_f1_arabert"].loc[p, PRIMARY].notna().all()]
    feats = feats.loc[ids]
    lab_bb = labels(piv, ids, "best_balance")
    lab_cost = labels(piv, ids, "cheapest")
    v1 = json.loads((SEL / "apcs_v1.json").read_text())
    v2 = json.loads((SEL / "apcs_v2.json").read_text())
    vc = json.loads((SEL / "apcs_cost.json").read_text())

    pol = {"APCS-1.0.0": pd.Series(apply_v1(v1, feats.token_count.to_numpy()), index=ids),
           "APCS-v2": pd.Series([walk(v2["tree"], feats.loc[p]) for p in ids], index=ids),
           "APCS-cost": pd.Series([walk(vc["tree"], feats.loc[p]) for p in ids], index=ids)}
    for c in COST:
        pol[f"always {c}"] = pd.Series(c, index=ids)
    rng = np.random.default_rng(SEED)
    pol["random selection"] = pd.Series(
        [PRIMARY[i] for i in rng.integers(len(PRIMARY), size=len(ids))], index=ids)

    def val(tab, choice):
        return np.array([tab.loc[p, c] if c in tab.columns else np.nan
                         for p, c in choice.items()], dtype=float)

    base_cost = val(piv["cost_usd"], pol["always noop@1.0"])
    qa_ids = [p for p in ids if cats[p] == "qa"]
    table = {}
    for name, ch in pol.items():
        f1, cost = val(piv["out_f1_arabert"], ch), val(piv["cost_usd"], ch)
        qa = val(piv["qa_contains"], ch.loc[qa_ids]) if qa_ids else np.array([])
        acc_ci = bootstrap_ci((ch == lab_bb).astype(float).to_numpy())
        table[name] = {
            "accuracy_best_balance": round(acc_ci[0], 4),
            "accuracy_ci95": [round(acc_ci[1], 4), round(acc_ci[2], 4)],
            "accuracy_cheapest_faithful": round(float((ch == lab_cost).mean()), 4),
            "mean_tcr": round(float(np.nanmean(val(piv["tcr"], ch))), 4),
            "mean_out_f1": round(float(np.nanmean(f1)), 4),
            "violation_rate": round(float(np.nanmean(f1 < TAU)), 4),
            "cost_change_pct": round(100 * float(np.nansum(cost) / base_cost.sum() - 1), 2),
            "qa_correct": (round(float(np.nanmean(qa)), 4)
                           if len(qa) and not np.all(np.isnan(qa)) else None),
        }

    def mc(a, b, lab):
        ca = (pol[a] == lab).to_numpy()
        cb = (pol[b] == lab).to_numpy()
        ao, bo, p = mcnemar_exact(ca, cb)
        d, lo, hi = paired_bootstrap_ci(ca.astype(float), cb.astype(float))
        return {"diff": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)],
                "a_only": ao, "b_only": bo, "p": p}

    h = {"H1_APCSv2_vs_" + H1_COMPARATOR.replace(" ", "_"):
         mc("APCS-v2", H1_COMPARATOR, lab_bb),
         "H2_APCSv2_vs_APCS-1.0.0": mc("APCS-v2", "APCS-1.0.0", lab_bb)}
    dc = val(piv["cost_usd"], pol["APCS-cost"]) - base_cost
    est, lo, hi = bootstrap_ci(dc)
    h["H3_APCScost_vs_never_compress"] = {
        "mean_cost_diff_usd_per_prompt": est, "ci95": [lo, hi],
        "cost_change_pct": table["APCS-cost"]["cost_change_pct"],
        "p": float(stats.wilcoxon(dc, alternative="less").pvalue)}
    # Holm correction over the three pre-registered hypotheses
    order = sorted(h, key=lambda k: h[k]["p"])
    running = 0.0
    for i, k in enumerate(order):
        adj = min(1.0, (len(order) - i) * h[k]["p"])
        running = max(running, adj)
        h[k]["p_holm"] = running
        # H1/H2: Holm-adjusted McNemar p < 0.05 and APCS-v2 ahead.
        # H3: Holm-adjusted one-sided Wilcoxon p < 0.05 and total cost lower.
        h[k]["supported"] = bool(running < 0.05 and (
            h[k]["diff"] > 0 if "diff" in h[k] else h[k]["cost_change_pct"] < 0))

    secondary = {name: mc("APCS-v2", name, lab_bb) for name in pol
                 if name not in ("APCS-v2",)}
    by_cat = {}
    for c in sorted(set(cats[p] for p in ids)):
        cid = [p for p in ids if cats[p] == c]
        by_cat[c] = {n: round(float((pol[n].loc[cid] == lab_bb.loc[cid]).mean()), 4)
                     for n in ["APCS-1.0.0", "APCS-v2", H1_COMPARATOR]}
    report = {"tau": TAU, "n_prompts": len(ids), "n_qa": len(qa_ids),
              "label_distribution_best_balance": lab_bb.value_counts(normalize=True)
              .round(4).to_dict(),
              "label_distribution_cheapest": lab_cost.value_counts(normalize=True)
              .round(4).to_dict(),
              "replication_APCS-1.0.0_accuracy": table["APCS-1.0.0"]["accuracy_best_balance"],
              "hypotheses": h, "policies": table,
              "secondary_APCSv2_vs_each": secondary,
              "accuracy_by_category": by_cat}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "exam_evaluation.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"n = {len(ids)} prompts | tau {TAU}")
    for k, v in h.items():
        print(f"  {k}: p={v['p']:.3g} holm={v['p_holm']:.3g} supported={v['supported']} | "
              + (f"diff {v['diff']:+.3f} {v['ci95']}" if "diff" in v
                 else f"cost {v['cost_change_pct']:+.1f}%"))
    for n in ["APCS-1.0.0", "APCS-v2", "APCS-cost", H1_COMPARATOR,
              "always noop@1.0", "random selection"]:
        t = table[n]
        print(f"  {n:28s} acc {t['accuracy_best_balance']:.3f} {t['accuracy_ci95']} | "
              f"TCR {t['mean_tcr']:.3f} | F1 {t['mean_out_f1']:.3f} | "
              f"viol {t['violation_rate']:.1%} | cost {t['cost_change_pct']:+.1f}% | "
              f"QA {t['qa_correct']}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-check", action="store_true")
    ap.add_argument("--outdir", default=str(RESULTS))
    args = ap.parse_args()
    if args.dev_check:
        fr = pd.read_csv(EXP10 / "final_results.csv")
        dev = fr[(fr.origin == "v2") & (fr.split == "dev")].copy()
        prot = pd.read_csv(RESULTS / "dev_protected.csv")
        df = pd.concat([dev, prot], ignore_index=True)
        price = json.loads((EXP10 / "api_config.json").read_text())["price_per_M"]
        df["cost_usd"] = (df.resp_prompt_tokens * price["in"]
                          + df.resp_completion_tokens * price["out"]) / 1e6
        df["qa_contains"] = np.nan
        sys.path.insert(0, str(ROOT))
        from design_selectors import load
        _, X, cats = load()
        evaluate(tables(df), X, cats, Path(args.outdir))
        return
    df = pd.read_csv(RESULTS / "exam_results.csv")
    feats = pd.read_csv(RESULTS / "exam_features.csv").set_index("prompt_id")
    cats = feats.task_category
    evaluate(tables(df), feats, cats, Path(args.outdir))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
