"""Exp 11c — Design APCS-v2 (primary) and APCS-cost (extension) on v2 DEV.

Both are shallow decision trees (readable as if/then rules) over the APCS
feature vector plus task category and a has-length-instruction flag.
Complexity (depth, minimum leaf size) is chosen by 10-fold stratified
cross-validation inside the 800 dev prompts. The v2 test split and the exam
set are not read.

APCS-v2   label = best balance (proposal definition): the candidate with the
          largest TCR whose output F1 >= tau, else no compression.
          Candidates: no compression, LLMLingua-2 @ 0.7 / 0.5 / 0.3
          (same as APCS 1.0.0, so the comparison isolates the selector).
APCS-cost label = cheapest measured call whose output F1 >= tau (ties ->
          larger TCR). Candidates add LLMLingua-2-protected @ 0.7/0.5/0.3 and
          LLMLingua @ 0.7/0.5/0.3. No compression is always eligible.

Outputs: results/selectors/{apcs_v2,apcs_cost}.json (tree + rules text),
         results/selector_design.json (CV evidence)
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text

from protect import has_length_instruction

ROOT = Path(__file__).resolve().parent
EXP10 = ROOT.parent / "10_final_benchmark" / "results"
RESULTS = ROOT / "results"
OUT = RESULTS / "selectors"
SEED = 42
TAU = json.loads((EXP10 / "analysis_10.json").read_text())["tau"]["arabert"]["adopted"]
PRICE = json.loads((EXP10 / "api_config.json").read_text())["price_per_M"]

PRIMARY = ["noop@1.0", "llmlingua2@0.7", "llmlingua2@0.5", "llmlingua2@0.3"]
COST = PRIMARY + [f"llmlingua2_protected@{r}" for r in (0.7, 0.5, 0.3)] \
    + [f"llmlingua_qwen@{r}" for r in (0.7, 0.5, 0.3)]
CATS = ["instruction", "summarisation", "qa", "creative"]
FEATURES = ["token_count", "structural_complexity", "fragmentation_ratio",
            "morphological_density", "has_length_instruction"] \
    + [f"cat_{c}" for c in CATS]
DEPTHS = [1, 2, 3, 4, 5]
LEAVES = [10, 20, 40]


def load():
    fr = pd.read_csv(EXP10 / "final_results.csv")
    dev = fr[(fr.origin == "v2") & (fr.split == "dev")].copy()
    prot = pd.read_csv(RESULTS / "dev_protected.csv")
    rows = pd.concat([dev, prot], ignore_index=True)
    rows["cand"] = rows.method + "@" + rows.target_rate.astype(str)
    rows["cost_usd"] = (rows.resp_prompt_tokens * PRICE["in"]
                        + rows.resp_completion_tokens * PRICE["out"]) / 1e6
    piv = {v: rows.pivot_table(index="prompt_id", columns="cand", values=v)
           for v in ["tcr", "out_f1_arabert", "cost_usd"]}
    feats = pd.read_csv(EXP10 / "features_final.csv").set_index("prompt_id")
    noop = dev[dev.method == "noop"].set_index("prompt_id")
    X = feats.loc[noop.index, FEATURES[:4]].copy()
    X["has_length_instruction"] = noop.compressed.map(has_length_instruction).astype(int)
    for c in CATS:
        X[f"cat_{c}"] = (noop.category == c).astype(int)
    return piv, X[FEATURES], noop.category


def label_best_balance(piv, ids):
    tcr, f1 = piv["tcr"], piv["out_f1_arabert"]
    out = []
    for p in ids:
        ok = [c for c in PRIMARY if f1.loc[p, c] >= TAU]
        out.append(max(ok, key=lambda c: tcr.loc[p, c]) if ok else "noop@1.0")
    return np.array(out)


def label_cheapest(piv, ids):
    tcr, f1, cost = piv["tcr"], piv["out_f1_arabert"], piv["cost_usd"]
    out = []
    for p in ids:
        ok = [c for c in COST if c in f1.columns and pd.notna(f1.loc[p, c])
              and f1.loc[p, c] >= TAU]
        ok = ok or ["noop@1.0"]
        out.append(min(ok, key=lambda c: (cost.loc[p, c], -tcr.loc[p, c])))
    return np.array(out)


def v1_rule(tok, T1, T2, r_mid, r_long):
    return np.where(tok < T1, "noop@1.0", np.where(
        tok >= T2, f"llmlingua2@{r_long}", f"llmlingua2@{r_mid}"))


def fit_v1(tok, y):
    best, acc = None, -1
    for T1, T2, rm, rl in itertools.product([40, 60, 80, 100, 120, 150, 200],
                                            [200, 250, 300, 350, 400, 500],
                                            [0.7, 0.5, 0.3], [0.7, 0.5, 0.3]):
        if T2 > T1:
            a = np.mean(v1_rule(tok, T1, T2, rm, rl) == y)
            if a > acc:
                acc, best = a, dict(T1=T1, T2=T2, r_mid=rm, r_long=rl)
    return best


def realised(piv, ids, choice):
    f1 = np.array([piv["out_f1_arabert"].loc[p, c] for p, c in zip(ids, choice)])
    cost = np.array([piv["cost_usd"].loc[p, c] for p, c in zip(ids, choice)])
    base = np.array([piv["cost_usd"].loc[p, "noop@1.0"] for p in ids])
    tcr = np.array([piv["tcr"].loc[p, c] for p, c in zip(ids, choice)])
    return {"cost_change_pct": 100 * (cost.sum() / base.sum() - 1),
            "violation_rate": float(np.mean(f1 < TAU)),
            "mean_tcr": float(np.mean(tcr))}


def cross_validate(X, y, piv, cats):
    ids = np.array(X.index)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    res = {}
    for d, m in itertools.product(DEPTHS, LEAVES):
        acc, picks = [], np.empty(len(y), dtype=object)
        for tr, te in skf.split(X, y):
            t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=m,
                                       random_state=SEED).fit(X.iloc[tr], y[tr])
            picks[te] = t.predict(X.iloc[te])
            acc.append(np.mean(picks[te] == y[te]))
        res[(d, m)] = {"cv_accuracy": float(np.mean(acc)),
                       "cv_sd": float(np.std(acc)),
                       **realised(piv, ids, picks)}
    # references under the same folds
    ref = {"v1_rule": [], "fixed": {}}
    tok = X.token_count.to_numpy()
    for tr, te in skf.split(X, y):
        rule = fit_v1(tok[tr], y[tr])
        ref["v1_rule"].append(np.mean(v1_rule(tok[te], **rule) == y[te]))
    for c in sorted(set(y)):
        ref["fixed"][c] = float(np.mean(y == c))
    return res, {"v1_rule_cv_accuracy": float(np.mean(ref["v1_rule"])),
                 "fixed_strategy_accuracy": ref["fixed"]}


def pick(res):
    best = max(res.values(), key=lambda v: v["cv_accuracy"])["cv_accuracy"]
    # simplest config within 1 point of the best (prefer shallow, large leaves)
    cands = [k for k, v in res.items() if v["cv_accuracy"] >= best - 0.01]
    return sorted(cands, key=lambda k: (k[0], -k[1]))[0]


def tree_json(t, names):
    tr = t.tree_

    def node(i):
        if tr.children_left[i] == -1:
            return {"leaf": t.classes_[int(np.argmax(tr.value[i][0]))],
                    "n": int(tr.n_node_samples[i])}
        return {"feature": names[tr.feature[i]],
                "threshold": round(float(tr.threshold[i]), 4),
                "le": node(tr.children_left[i]), "gt": node(tr.children_right[i])}
    return node(0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    piv, X, cats = load()
    ids = list(X.index)
    design = {"tau": TAU, "n_dev": len(ids), "features": FEATURES}
    for name, labeller, cands in [("apcs_v2", label_best_balance, PRIMARY),
                                  ("apcs_cost", label_cheapest, COST)]:
        y = labeller(piv, ids)
        res, ref = cross_validate(X, y, piv, cats)
        d, m = pick(res)
        tree = DecisionTreeClassifier(max_depth=d, min_samples_leaf=m,
                                      random_state=SEED).fit(X, y)
        rules = export_text(tree, feature_names=FEATURES)
        spec = {"name": name, "candidates": cands, "tau": TAU,
                "label": "best balance" if name == "apcs_v2"
                         else "cheapest faithful (measured cost)",
                "max_depth": d, "min_samples_leaf": m, "features": FEATURES,
                "tree": tree_json(tree, FEATURES), "rules_text": rules,
                "trained_on": "AraPromptBench v2 dev (800 prompts)"}
        (OUT / f"{name}.json").write_text(json.dumps(spec, indent=2,
                                                     ensure_ascii=False))
        design[name] = {
            "label_distribution": pd.Series(y).value_counts(normalize=True)
            .round(4).to_dict(),
            "chosen": {"max_depth": d, "min_samples_leaf": m,
                       **{k: round(v, 4) for k, v in res[(d, m)].items()}},
            "cv_grid": {f"d{k[0]}_leaf{k[1]}": {kk: round(vv, 4)
                                                 for kk, vv in v.items()}
                        for k, v in res.items()},
            **ref}
        print(f"\n=== {name} (tau {TAU}) | labels "
              f"{design[name]['label_distribution']}")
        print(f"chosen depth {d}, leaf {m}: CV accuracy "
              f"{res[(d, m)]['cv_accuracy']:.3f} | v1-style rule CV "
              f"{ref['v1_rule_cv_accuracy']:.3f} | best fixed "
              f"{max(ref['fixed_strategy_accuracy'].values()):.3f}")
        print(f"CV realised: cost {res[(d, m)]['cost_change_pct']:+.1f}% vs none, "
              f"violations {res[(d, m)]['violation_rate']:.1%}, "
              f"TCR {res[(d, m)]['mean_tcr']:.3f}")
        print(rules)
    (RESULTS / "selector_design.json").write_text(json.dumps(design, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
