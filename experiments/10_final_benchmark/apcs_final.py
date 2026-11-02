"""Exp 10d — Final APCS calibration (v2 dev) + ONE-SHOT held-out evaluation.

Calibration (identical procedure to Exp 06, on the 800 dev prompts):
  best-balance label = max TCR among candidates with out-F1 >= tau, else noop;
  candidates {noop, llmlingua2 @ 0.7/0.5/0.3}; grid-search the token-count
  rule; global rule shipped to apcs/apcs/rules_default.json.

One-shot evaluation on the 200 test prompts — the only place test rows are
read. Baselines per SDR: no compression, always-LLMLingua, always-LLMLingua-2
(each rate), random selection. Reported per policy:
  accuracy (primary labels) and accuracy under extended labels that also
  admit LLMLingua-1 candidates; mean TCR; mean out-F1; fidelity-violation
  rate; measured API cost and saving vs no compression (USD, real token
  usage of the chosen variant's response); SDR Pareto-hit rate (footnote
  metric). Paired bootstrap 95% CIs + exact McNemar vs APCS, confusion
  matrix, per-category accuracy, and an mBERT sensitivity re-run with its own
  ceiling-derived tau.

--dry-run: evaluates on a stratified 20% pseudo-test drawn from DEV; real
test rows are dropped before anything else; package rules are not written.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stats_utils import bootstrap_ci, mcnemar_exact, paired_bootstrap_ci

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PROJECT = ROOT.parent.parent
PKG_RULES = PROJECT / "apcs" / "apcs" / "rules_default.json"

PRIMARY = ["noop@1.0", "llmlingua2@0.7", "llmlingua2@0.5", "llmlingua2@0.3"]
LLMLINGUA1 = ["llmlingua_qwen@0.7", "llmlingua_qwen@0.5", "llmlingua_qwen@0.3"]
EXTENDED = PRIMARY + LLMLINGUA1
T1_GRID = [40, 60, 80, 100, 120, 150, 200]
T2_GRID = [200, 250, 300, 350, 400, 500]
RATE_GRID = [0.7, 0.5, 0.3]
SEED = 42


def load(indir, dry_run):
    df = pd.read_csv(indir / "final_results.csv")
    df = df[df.origin == "v2"].copy()
    df["cand"] = df.method + "@" + df.target_rate.astype(str)
    cfg_path = RESULTS / "api_config.json"
    price = (json.loads(cfg_path.read_text())["price_per_M"] if cfg_path.exists()
             else {"in": 0.20, "out": 1.20})
    df["cost_usd"] = (df.resp_prompt_tokens * price["in"]
                      + df.resp_completion_tokens * price["out"]) / 1e6
    piv = {v: df.pivot_table(index="prompt_id", columns="cand", values=v)
           for v in ["tcr", "out_f1_arabert", "out_f1_mbert", "cost_usd"]}
    info = df.groupby("prompt_id").agg(tok=("orig_tokens", "first"),
                                       cat=("category", "first"),
                                       split=("split", "first"))
    # a prompt is evaluable only if every primary candidate was measured
    complete = piv["out_f1_arabert"].reindex(columns=PRIMARY).notna().all(axis=1)
    info = info[info.index.isin(complete[complete].index)]
    if dry_run:
        info = info[info.split == "dev"].copy()
        rng = np.random.default_rng(SEED)
        info["split"] = "dev"
        for _, g in info.groupby("cat"):
            pick = rng.choice(g.index, size=round(0.2 * len(g)), replace=False)
            info.loc[pick, "split"] = "test"
    return piv, info, price


def labels(ids, piv, f1col, tau, cands):
    tcr, f1 = piv["tcr"], piv[f1col]
    out = {}
    for p in ids:
        ok = [c for c in cands if c in f1.columns and pd.notna(f1.loc[p, c])
              and f1.loc[p, c] >= tau]
        out[p] = max(ok, key=lambda c: tcr.loc[p, c]) if ok else "noop@1.0"
    return pd.Series(out)


def rule_pred(tok, T1, T2, r_mid, r_long):
    return np.where(tok < T1, "noop@1.0",
                    np.where(tok >= T2, f"llmlingua2@{r_long}",
                             f"llmlingua2@{r_mid}"))


def grid_search(ids, lab, tok):
    t = tok.loc[ids].to_numpy()
    y = lab.loc[ids].to_numpy()
    best, acc = None, -1.0
    for T1, T2, rm, rl in itertools.product(T1_GRID, T2_GRID, RATE_GRID,
                                            RATE_GRID):
        if T2 <= T1:
            continue
        a = float(np.mean(rule_pred(t, T1, T2, rm, rl) == y))
        if a > acc:
            acc, best = a, dict(T1=T1, T2=T2, r_mid=rm, r_long=rl)
    return best, acc


def pareto_flags(piv, ids):
    tcr, f1 = piv["tcr"], piv["out_f1_arabert"]
    flags = {}
    for p in ids:
        pts = pd.DataFrame({"t": tcr.loc[p], "f": f1.loc[p]}).dropna()
        nd = set()
        for c, (t, f) in pts.iterrows():
            dom = ((pts.t >= t) & (pts.f >= f) & ((pts.t > t) | (pts.f > f))).any()
            if not dom:
                nd.add(c)
        flags[p] = nd
    return flags


def evaluate(policies, ids, lab, lab_ext, piv, f1col, tau, pflags):
    tcr, f1, cost = piv["tcr"], piv[f1col], piv["cost_usd"]
    base_cost = cost.loc[ids, "noop@1.0"].sum()
    rows = {}
    for name, pol in policies.items():
        ch = pol.loc[ids]
        pick = lambda tab: np.array([tab.loc[p, ch[p]] for p in ids])
        c = pick(cost)
        rows[name] = {
            "accuracy": round(float(np.mean(ch == lab.loc[ids])), 4),
            "accuracy_extended": round(float(np.mean(ch == lab_ext.loc[ids])), 4),
            "mean_tcr": round(float(np.nanmean(pick(tcr))), 4),
            "mean_out_f1": round(float(np.nanmean(pick(f1))), 4),
            "violation_rate": round(float(np.nanmean(pick(f1) < tau)), 4),
            "cost_usd": round(float(np.nansum(c)), 6),
            "saving_vs_noop_pct": round(100 * float(1 - np.nansum(c) / base_cost), 2),
            "pareto_hit": round(float(np.mean([ch[p] in pflags[p] for p in ids])), 4),
        }
    return rows


def run_scorer(piv, info, f1col, tau, dev_ids, test_ids, pflags, tag):
    dev_lab = labels(dev_ids, piv, f1col, tau, PRIMARY)
    rule, dev_acc = grid_search(dev_ids, dev_lab, info.tok)
    cat_gain = 0.0
    per_cat = {}
    for c, g in info.loc[dev_ids].groupby("cat"):
        r_c, a_c = grid_search(list(g.index), dev_lab, info.tok)
        per_cat[c] = {"rule": r_c, "dev_accuracy": round(a_c, 4)}
        cat_gain += a_c * len(g)
    cat_gain = cat_gain / len(dev_ids) - dev_acc

    test_lab = labels(test_ids, piv, f1col, tau, PRIMARY)
    test_lab_ext = labels(test_ids, piv, f1col, tau, EXTENDED)
    apcs = pd.Series(rule_pred(info.tok.loc[test_ids].to_numpy(), **rule),
                     index=test_ids)
    rng = np.random.default_rng(SEED)
    policies = {"APCS": apcs}
    for c in PRIMARY + LLMLINGUA1:
        policies[f"always {c}"] = pd.Series(c, index=test_ids)
    policies["random selection"] = pd.Series(
        [PRIMARY[i] for i in rng.integers(len(PRIMARY), size=len(test_ids))],
        index=test_ids)
    table = evaluate(policies, test_ids, test_lab, test_lab_ext, piv, f1col,
                     tau, pflags)

    correct = {n: (p.loc[test_ids] == test_lab.loc[test_ids]).to_numpy()
               for n, p in policies.items()}
    correct_ext = {n: (p.loc[test_ids] == test_lab_ext.loc[test_ids]).to_numpy()
                   for n, p in policies.items()}
    comparisons = {}
    for name in policies:
        if name == "APCS":
            continue
        # an always-LLMLingua policy can only be right under labels that admit
        # LLMLingua candidates, so it is compared on the extended label set
        ext = "llmlingua_qwen" in name
        ca, cb = ((correct_ext["APCS"], correct_ext[name]) if ext
                  else (correct["APCS"], correct[name]))
        a_only, b_only, p = mcnemar_exact(ca, cb)
        d, lo, hi = paired_bootstrap_ci(ca.astype(float), cb.astype(float))
        comparisons[name] = {"label_set": "extended" if ext else "primary",
                             "acc_diff": round(d, 4),
                             "ci95": [round(lo, 4), round(hi, 4)],
                             "apcs_only_correct": a_only,
                             "baseline_only_correct": b_only,
                             "mcnemar_p": p}
    acc_ci = bootstrap_ci(correct["APCS"].astype(float))
    return {
        "scorer": tag, "tau": tau, "rule": rule,
        "dev_accuracy": round(dev_acc, 4),
        "per_category_rules_gain": round(float(cat_gain), 4),
        "per_category_rules": per_cat,
        "dev_label_distribution": dev_lab.value_counts(normalize=True)
        .round(4).to_dict(),
        "test_label_distribution": test_lab.value_counts(normalize=True)
        .round(4).to_dict(),
        "test_policies": table,
        "apcs_accuracy_ci95": [round(acc_ci[1], 4), round(acc_ci[2], 4)],
        "apcs_vs_baselines": comparisons,
        "apcs_beats_all_baselines": all(
            v["acc_diff"] > 0 for v in comparisons.values()),
        "apcs_significantly_beats": [
            k for k, v in comparisons.items()
            if v["ci95"][0] > 0 and v["mcnemar_p"] < 0.05],
        "confusion_matrix": pd.crosstab(
            test_lab.loc[test_ids].rename("label"),
            apcs.rename("apcs")).reindex(index=PRIMARY, columns=PRIMARY,
                                          fill_value=0).to_dict(),
        "test_accuracy_by_category": {
            c: dict(zip(["acc", "lo", "hi"],
                        [round(x, 4) for x in bootstrap_ci(
                            correct["APCS"][np.isin(test_ids, list(g.index))]
                            .astype(float))]),
                    n=len(g))
            for c, g in info.loc[test_ids].groupby("cat")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=str(RESULTS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    indir = Path(args.indir)
    analysis = json.loads((indir / "analysis_10.json").read_text())
    piv, info, price = load(indir, args.dry_run)
    dev_ids = list(info[info.split == "dev"].index)
    test_ids = list(info[info.split == "test"].index)
    print(f"dev {len(dev_ids)} | test {len(test_ids)}"
          f"{' (PSEUDO-TEST from dev)' if args.dry_run else ''}")
    pflags = pareto_flags(piv, test_ids)

    report = {"dry_run": args.dry_run, "price_per_M": price,
              "n_dev": len(dev_ids), "n_test": len(test_ids)}
    report["primary_arabert"] = run_scorer(
        piv, info, "out_f1_arabert", analysis["tau"]["arabert"]["adopted"],
        dev_ids, test_ids, pflags, "arabert")
    report["sensitivity_mbert"] = run_scorer(
        piv, info, "out_f1_mbert", analysis["tau"]["mbert"]["adopted"],
        dev_ids, test_ids, pflags, "mbert")

    if not args.dry_run:
        rule = report["primary_arabert"]["rule"]
        PKG_RULES.write_text(json.dumps({
            "version": "1.0.0-final",
            "calibrated_on": "AraPromptBench v2 dev split (800 prompts)",
            "llm_under_test": json.loads((RESULTS / "api_config.json")
                                         .read_text())["model"],
            "fidelity_threshold_tau": report["primary_arabert"]["tau"],
            "decision_feature": "token_count (cl100k_base)",
            "rules": [
                {"if": f"token_count < {rule['T1']}",
                 "then": {"method": "none", "rate": 1.0}},
                {"if": f"token_count >= {rule['T2']}",
                 "then": {"method": "llmlingua2", "rate": rule["r_long"]}},
                {"if": "otherwise",
                 "then": {"method": "llmlingua2", "rate": rule["r_mid"]}}],
            "thresholds": rule}, indent=2), encoding="utf-8")

    (indir / "apcs_final.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    p = report["primary_arabert"]
    print(f"rule {p['rule']} | dev acc {p['dev_accuracy']} | "
          f"test acc {p['test_policies']['APCS']['accuracy']} "
          f"CI {p['apcs_accuracy_ci95']}")
    for k, v in p["apcs_vs_baselines"].items():
        print(f"  vs {k:26s} [{v['label_set']:8s}] diff {v['acc_diff']:+.3f} "
              f"{v['ci95']} p={v['mcnemar_p']:.2g}")
    print(f"mBERT: beats all baselines = "
          f"{report['sensitivity_mbert']['apcs_beats_all_baselines']}, "
          f"rule {report['sensitivity_mbert']['rule']}")
    print(f"saved apcs_final.json to {indir}"
          f"{'' if args.dry_run else ' + package rules v1.0.0-final'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
