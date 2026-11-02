"""Experiment 06 — APCS rule calibration (RQ3, pilot).

Derives the rule-based selector's thresholds from the 500-prompt pilot's
measured outcomes (Exp 05 pareto_labels.csv), then evaluates the calibrated
policy against baselines. Because every (prompt, method, rate) outcome was
measured in Exps 03/04, a policy is evaluated by lookup — no new runs.

Ground-truth label ("best balance", per proposal): the candidate with maximum
TCR among those with output-level F1 (AraBERT) >= tau; NoCompression if none
qualifies. The SDR's Pareto-hit metric is reported alongside but is degenerate
(noop is always Pareto-optimal), so label accuracy is the primary metric.

Candidate set for recommendations: noop, llmlingua2 @ {0.7, 0.5, 0.3}.
(LLMLingua-1 is never uniquely optimal like-for-like — Exp 04 Finding 2;
random deletion is dominated — Exp 04 Finding 1.)

Rule template (from Exp 05 findings — token_count is the dominant feature):
    if token_count <  T1:  noop
    elif token_count >= T2: (llmlingua2, r_long)
    else:                   (llmlingua2, r_mid)
Grid-searched: T1, T2, r_mid, r_long. A per-category variant is also searched
to test whether category-specific thresholds add accuracy.

Outputs: results/calibration.json (chosen rules + all metrics),
         results/policy_comparison.csv, ../../apcs/apcs/rules_default.json
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TAUS = [0.65, 0.70, 0.75]
TAU_STAR = 0.70  # working threshold for the shipped rules (provisional
                 # pending repeat-call ceiling measurement, Exp 04 caveat)
T1_GRID = [40, 60, 80, 100, 120, 150]
T2_GRID = [200, 250, 300, 350, 400, 500]
RATE_GRID = [0.7, 0.5, 0.3]

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
LABELS = ROOT.parent / "05_feature_analysis" / "results" / "pareto_labels.csv"
PKG_RULES = ROOT.parent.parent / "apcs" / "apcs" / "rules_default.json"

CANDIDATES = ["noop@1.0", "llmlingua2@0.7", "llmlingua2@0.5", "llmlingua2@0.3"]


def build_tables(df: pd.DataFrame):
    """per-prompt outcome lookup keyed by 'method@rate' strings."""
    df = df.copy()
    df["cand"] = df.method + "@" + df.target_rate.astype(str)
    sub = df[df.cand.isin(CANDIDATES)].copy()
    piv_tcr = sub.pivot_table(index="prompt_id", columns="cand", values="tcr")
    piv_f1 = sub.pivot_table(index="prompt_id", columns="cand",
                             values="out_f1_arabert")
    pareto = sub.pivot_table(index="prompt_id", columns="cand",
                             values="is_pareto")
    feats = sub.groupby("prompt_id").agg(
        token_count=("token_count", "first"),
        structural_complexity=("structural_complexity", "first"),
        task_category=("task_category", "first")).reset_index()
    return piv_tcr, piv_f1, pareto, feats.set_index("prompt_id")


def utility_label(piv_tcr, piv_f1, tau):
    """best-balance label per prompt: max TCR s.t. F1 >= tau, else noop."""
    labels = {}
    for pid in piv_tcr.index:
        ok = [c for c in CANDIDATES if piv_f1.loc[pid, c] >= tau]
        labels[pid] = max(ok, key=lambda c: piv_tcr.loc[pid, c]) if ok \
            else "noop@1.0"
    return pd.Series(labels)


def apply_rule(tok, T1, T2, r_mid, r_long):
    if tok < T1:
        return "noop@1.0"
    if tok >= T2:
        return f"llmlingua2@{r_long}"
    return f"llmlingua2@{r_mid}"


def policy_metrics(choice: pd.Series, labels, piv_tcr, piv_f1, pareto, tau):
    acc = float((choice == labels).mean())
    tcr = float(np.mean([piv_tcr.loc[p, c] for p, c in choice.items()]))
    f1 = float(np.mean([piv_f1.loc[p, c] for p, c in choice.items()]))
    viol = float(np.mean([piv_f1.loc[p, c] < tau for p, c in choice.items()]))
    phit = float(np.mean([pareto.loc[p, c] for p, c in choice.items()]))
    return dict(label_accuracy=round(acc, 4), mean_tcr=round(tcr, 4),
                mean_out_f1=round(f1, 4),
                fidelity_violation_rate=round(viol, 4),
                sdr_pareto_hit_rate=round(phit, 4))


def search(feats, labels, per_category=False):
    grids = itertools.product(T1_GRID, T2_GRID, RATE_GRID, RATE_GRID)
    best, best_acc = None, -1
    for T1, T2, r_mid, r_long in grids:
        if T2 <= T1:
            continue
        choice = feats.token_count.apply(
            lambda t: apply_rule(t, T1, T2, r_mid, r_long))
        acc = (choice == labels).mean()
        if acc > best_acc:
            best_acc, best = acc, dict(T1=T1, T2=T2, r_mid=r_mid,
                                       r_long=r_long)
    if not per_category:
        return best, best_acc
    # per-category: independent params per category
    cat_rules, accs = {}, []
    for cat, grp in feats.groupby("task_category"):
        sub_labels = labels.loc[grp.index]
        b, a = search(grp, sub_labels, per_category=False)
        cat_rules[cat] = b
        accs.append(a * len(grp))
    return cat_rules, sum(accs) / len(feats)


def main() -> None:
    df = pd.read_csv(LABELS)
    piv_tcr, piv_f1, pareto, feats = build_tables(df)
    print(f"{len(feats)} prompts, candidates: {CANDIDATES}")

    report = {"taus": {}}
    for tau in TAUS:
        labels = utility_label(piv_tcr, piv_f1, tau)
        dist = labels.value_counts(normalize=True).round(3)
        best, acc = search(feats, labels)
        report["taus"][str(tau)] = {
            "label_distribution": {str(k): float(v) for k, v in dist.items()},
            "best_rule": best, "rule_accuracy": round(float(acc), 4)}
        print(f"\ntau={tau}: label dist {dict(dist)}")
        print(f"  best global rule {best}, accuracy {acc:.3f}")

    # ---- final calibration at TAU_STAR ------------------------------------
    labels = utility_label(piv_tcr, piv_f1, TAU_STAR)
    best, acc = search(feats, labels)
    cat_rules, cat_acc = search(feats, labels, per_category=True)
    print(f"\n=== tau*={TAU_STAR} ===")
    print(f"global rule: {best} -> accuracy {acc:.3f}")
    print(f"per-category rules -> accuracy {cat_acc:.3f} "
          f"(+{cat_acc-acc:.3f} vs global)")

    # ---- policy comparison table ------------------------------------------
    rows = {}
    choice = feats.token_count.apply(lambda t: apply_rule(t, **best))
    rows["APCS (global rule)"] = policy_metrics(
        choice, labels, piv_tcr, piv_f1, pareto, TAU_STAR)
    cat_choice = pd.Series({p: apply_rule(feats.loc[p].token_count,
                                          **cat_rules[feats.loc[p].task_category])
                            for p in feats.index})
    rows["APCS (per-category)"] = policy_metrics(
        cat_choice, labels, piv_tcr, piv_f1, pareto, TAU_STAR)
    for cand in CANDIDATES:
        fixed = pd.Series({p: cand for p in feats.index})
        rows[f"always {cand}"] = policy_metrics(fixed, labels, piv_tcr,
                                                piv_f1, pareto, TAU_STAR)
    rng = np.random.default_rng(42)
    rand = pd.Series({p: CANDIDATES[rng.integers(len(CANDIDATES))]
                      for p in feats.index})
    rows["random selection"] = policy_metrics(rand, labels, piv_tcr, piv_f1,
                                              pareto, TAU_STAR)
    oracle = policy_metrics(labels, labels, piv_tcr, piv_f1, pareto, TAU_STAR)
    rows["oracle (label itself)"] = oracle

    comp = pd.DataFrame(rows).T
    comp.to_csv(RESULTS / "policy_comparison.csv", encoding="utf-8-sig")
    print("\n=== Policy comparison (tau*=0.70, in-sample pilot) ===")
    print(comp.to_string())

    report["tau_star"] = TAU_STAR
    report["global_rule"] = best
    report["per_category_rules"] = cat_rules
    report["policy_comparison"] = {k: v for k, v in rows.items()}
    (RESULTS / "calibration.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    # ---- ship rules to the package ----------------------------------------
    PKG_RULES.parent.mkdir(parents=True, exist_ok=True)
    shipped = {
        "version": "0.1.0-pilot",
        "calibrated_on": "AraPromptBench v1.1.0 (500-prompt pilot, dev-only)",
        "fidelity_threshold_tau": TAU_STAR,
        "decision_feature": "token_count (cl100k_base)",
        "rules": [
            {"if": f"token_count < {best['T1']}",
             "then": {"method": "none", "rate": 1.0}},
            {"if": f"token_count >= {best['T2']}",
             "then": {"method": "llmlingua2", "rate": best["r_long"]}},
            {"if": "otherwise",
             "then": {"method": "llmlingua2", "rate": best["r_mid"]}},
        ],
        "thresholds": best,
    }
    PKG_RULES.write_text(json.dumps(shipped, indent=2), encoding="utf-8")
    print(f"\nSaved: calibration.json, policy_comparison.csv, {PKG_RULES}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
