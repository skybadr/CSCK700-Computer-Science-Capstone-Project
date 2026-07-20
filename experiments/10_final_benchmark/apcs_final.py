"""Exp 10d — Final APCS calibration (v2 dev) + ONE-SHOT held-out evaluation.

Procedure identical to Exp 06, executed on the final dataset:
  1. best-balance labels on v2 DEV (tau from analysis_10.json)
  2. grid-search rule template on DEV -> ships apcs/apcs/rules_default.json v2
  3. per-category variant compared on DEV (kept only if it beats global
     meaningfully)
  4. THE ONE-SHOT: frozen rule applied to the 200 TEST prompts; accuracy vs
     baselines, McNemar exact tests, policy metrics. Test rows are read here
     and nowhere else.

Output: results/apcs_final.json, updated package rules.
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PROJECT = ROOT.parent.parent
PKG_RULES = PROJECT / "apcs" / "apcs" / "rules_default.json"

CANDS = ["noop@1.0", "llmlingua2@0.7", "llmlingua2@0.5", "llmlingua2@0.3"]
T1_GRID = [40, 60, 80, 100, 120, 150]
T2_GRID = [200, 250, 300, 350, 400, 500]
RATE_GRID = [0.7, 0.5, 0.3]


def tables(df):
    df = df.copy()
    df["cand"] = df.method + "@" + df.target_rate.astype(str)
    sub = df[df.cand.isin(CANDS)]
    tcr = sub.pivot_table(index="prompt_id", columns="cand", values="tcr")
    f1 = sub.pivot_table(index="prompt_id", columns="cand",
                         values="out_f1_arabert")
    tok = sub.groupby("prompt_id").orig_tokens.first()
    cat = sub.groupby("prompt_id").category.first()
    return tcr, f1, tok, cat


def labels_of(ids, tcr, f1, tau):
    out = {}
    for p in ids:
        ok = [c for c in CANDS if f1.loc[p, c] >= tau]
        out[p] = max(ok, key=lambda c: tcr.loc[p, c]) if ok else "noop@1.0"
    return out


def rule_fn(T1, T2, rm, rl):
    def f(t):
        if t < T1:
            return "noop@1.0"
        if t >= T2:
            return f"llmlingua2@{rl}"
        return f"llmlingua2@{rm}"
    return f


def grid_search(ids, lab, tok):
    toks = np.array([tok[p] for p in ids])
    labs = np.array([lab[p] for p in ids])
    best, acc = None, -1.0
    for T1, T2, rm, rl in itertools.product(T1_GRID, T2_GRID, RATE_GRID,
                                            RATE_GRID):
        if T2 <= T1:
            continue
        pred = np.where(toks < T1, "noop@1.0",
                        np.where(toks >= T2, f"llmlingua2@{rl}",
                                 f"llmlingua2@{rm}"))
        a = float(np.mean(pred == labs))
        if a > acc:
            acc, best = a, dict(T1=T1, T2=T2, r_mid=rm, r_long=rl)
    return best, acc


def metrics(choice, lab, tcr, f1, tau):
    ids = list(choice)
    acc = float(np.mean([choice[p] == lab[p] for p in ids]))
    mtcr = float(np.mean([tcr.loc[p, choice[p]] for p in ids]))
    mf1 = float(np.mean([f1.loc[p, choice[p]] for p in ids]))
    viol = float(np.mean([f1.loc[p, choice[p]] < tau for p in ids]))
    return dict(accuracy=round(acc, 4), mean_tcr=round(mtcr, 4),
                mean_out_f1=round(mf1, 4), violation_rate=round(viol, 4))


def mcnemar(choice_a, choice_b, lab):
    ids = list(lab)
    b = sum(choice_a[p] == lab[p] and choice_b[p] != lab[p] for p in ids)
    c = sum(choice_a[p] != lab[p] and choice_b[p] == lab[p] for p in ids)
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, float(p)


def main():
    tau = json.loads((RESULTS / "analysis_10.json").read_text())["tau"]["tau_adopted"]
    df = pd.read_csv(RESULTS / "final_results.csv")
    v2 = df[df.origin == "v2"]
    tcr, f1, tok, cat = tables(v2)
    split = v2.groupby("prompt_id")["split"].first()
    dev_ids = [p for p in tcr.index if split[p] == "dev"]
    test_ids = [p for p in tcr.index if split[p] == "test"]
    print(f"tau={tau} | dev {len(dev_ids)} | test {len(test_ids)}")

    dev_lab = labels_of(dev_ids, tcr, f1, tau)
    dist = pd.Series(dev_lab).value_counts(normalize=True).round(3)
    best, dev_acc = grid_search(dev_ids, dev_lab, tok)
    # per-category variant
    cat_rules, accs = {}, 0
    for c in sorted(set(cat[p] for p in dev_ids)):
        cids = [p for p in dev_ids if cat[p] == c]
        b, a = grid_search(cids, {p: dev_lab[p] for p in cids}, tok)
        cat_rules[c] = b
        accs += a * len(cids)
    cat_acc = accs / len(dev_ids)
    use_cat = cat_acc - dev_acc > 0.02
    print(f"dev: global {best} acc {dev_acc:.3f} | per-cat {cat_acc:.3f} "
          f"| use_cat={use_cat}")

    # ---- ship rules ----
    shipped = {"version": "1.0.0-final",
               "calibrated_on": "AraPromptBench v2 dev (800), final protocol",
               "fidelity_threshold_tau": tau,
               "decision_feature": "token_count (cl100k_base)",
               "thresholds": best,
               "per_category": cat_rules if use_cat else None,
               "rules": [
                   {"if": f"token_count < {best['T1']}",
                    "then": {"method": "none", "rate": 1.0}},
                   {"if": f"token_count >= {best['T2']}",
                    "then": {"method": "llmlingua2", "rate": best["r_long"]}},
                   {"if": "otherwise",
                    "then": {"method": "llmlingua2", "rate": best["r_mid"]}}]}
    PKG_RULES.write_text(json.dumps(shipped, indent=2), encoding="utf-8")

    # ---- ONE-SHOT held-out evaluation ----
    test_lab = labels_of(test_ids, tcr, f1, tau)
    rf = rule_fn(**best)
    apcs = {p: rf(tok[p]) for p in test_ids}
    policies = {"APCS": apcs}
    for cand in CANDS:
        policies[f"always {cand}"] = {p: cand for p in test_ids}
    rng = np.random.default_rng(42)
    policies["random selection"] = {p: CANDS[rng.integers(len(CANDS))]
                                    for p in test_ids}
    report = {"tau": tau, "dev_rule": best, "dev_accuracy": round(dev_acc, 4),
              "dev_label_distribution": {str(k): float(v)
                                         for k, v in dist.items()},
              "per_category_rules_used": use_cat,
              "test": {}}
    for name, pol in policies.items():
        report["test"][name] = metrics(pol, test_lab, tcr, f1, tau)
    report["mcnemar_vs_APCS"] = {}
    for name, pol in policies.items():
        if name == "APCS":
            continue
        b, c, p = mcnemar(apcs, pol, test_lab)
        report["mcnemar_vs_APCS"][name] = {"apcs_only_correct": b,
                                           "other_only_correct": c,
                                           "p": p}
    # per-category test accuracy for APCS
    report["test_APCS_by_category"] = {}
    for c in sorted(set(cat[p] for p in test_ids)):
        cids = [p for p in test_ids if cat[p] == c]
        report["test_APCS_by_category"][c] = round(float(np.mean(
            [apcs[p] == test_lab[p] for p in cids])), 4)

    (RESULTS / "apcs_final.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["test"], indent=2))
    print(json.dumps(report["mcnemar_vs_APCS"], indent=2))
    print("saved: apcs_final.json + package rules v1.0.0-final")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
