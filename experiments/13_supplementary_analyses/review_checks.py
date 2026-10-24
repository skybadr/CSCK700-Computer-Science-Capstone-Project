"""Exp 13 (supplementary, part 2) — checks raised by an independent pre-submission review.

All computed from stored results (no new API calls, no selector re-tuned; the
frozen exam selectors are only re-scored):
  L  billing tokeniser: billed input tokens vs cl100k_base and o200k_base counts;
     tokens per word and TCR agreement under both tokenisers
  M  H1 by category: APCS-v2 vs always LLMLingua-2 @ 0.5 with each category left out (exam)
  N  H3 on the total bill: ratio-of-sums bootstrap CI, zero differences, sign-flip
     permutation test on the mean (exam)
  O  tau sensitivity: exam best-balance labels re-derived at tau = 0.60 / 0.65 / 0.70
     and with the per-category 0.4th percentiles; H1/H2 re-tested with the frozen selectors
  P  RQ2 with the noise ceiling as covariate: OLS of LLMLingua-2 @ 0.5 output F1 with
     each prompt's repeat-call F1 added (dev)
  Q  per-category fixed-rate lookup (best rate per category on dev, applied to the exam)
     — exploratory, not pre-registered
  R  length-marker precision: prompts flagged only by "kalima" without a number before it
  S  APCS-cost choices and QA correctness on exam QA prompts
Output: results/review_checks.json
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import tiktoken

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
E10 = PROJECT / "experiments" / "10_final_benchmark" / "results"
E11D = PROJECT / "experiments" / "11_fresh_exam"
E11 = E11D / "results"
OUT = ROOT / "results"
sys.path.insert(0, str(PROJECT / "experiments" / "10_final_benchmark"))
sys.path.insert(0, str(E11D))
sys.path.insert(0, str(PROJECT / "apcs"))
from stats_utils import mcnemar_exact  # noqa: E402
from apcs.features import LENGTH_MARKERS  # noqa: E402
import evaluate_exam as ev  # noqa: E402

CL, O2 = tiktoken.get_encoding("cl100k_base"), tiktoken.get_encoding("o200k_base")
PRIMARY = ev.PRIMARY
CATS = ["instruction", "summarisation", "qa", "creative"]
R = {}
rng = np.random.default_rng(42)


def ntok(enc, t):
    return len(enc.encode(t or ""))


# L ---------------------------------------------------------------------------
fr = pd.read_csv(E10 / "final_results.csv")
v2 = fr[fr.origin == "v2"]
noop = v2[v2.method == "noop"].drop_duplicates("prompt_id").copy()
noop["cl"] = noop.compressed.map(lambda t: ntok(CL, t))
noop["o2"] = noop.compressed.map(lambda t: ntok(O2, t))
noop["words"] = noop.compressed.str.split().map(len)
gap = noop.resp_prompt_tokens - noop.o2
ll2 = v2[(v2.method == "llmlingua2") & v2.compressed.notna() & (v2.split == "dev")].copy()
o2_orig = noop.set_index("prompt_id").o2
ll2["tcr_o2"] = 1 - ll2.compressed.map(lambda t: ntok(O2, t)) / ll2.prompt_id.map(o2_orig)
fert = {k: noop[k] / noop.words for k in ("cl", "o2")}
R["L_tokeniser"] = {
    "n_prompts": int(len(noop)),
    "billed_minus_o200k": {"min": int(gap.min()), "max": int(gap.max()),
                           "share_equal_6": round(float((gap == 6).mean()), 4)},
    "corr_billed_o200k": round(float(np.corrcoef(noop.resp_prompt_tokens, noop.o2)[0, 1]), 5),
    "mean_tokens": {"cl100k": round(float(noop.cl.mean()), 1),
                    "o200k": round(float(noop.o2.mean()), 1),
                    "billed": round(float(noop.resp_prompt_tokens.mean()), 1)},
    "tokens_per_word": {k: {"mean": round(float(v.mean()), 2),
                            "p10": round(float(v.quantile(0.1)), 2),
                            "p90": round(float(v.quantile(0.9)), 2)}
                        for k, v in (("cl100k", fert["cl"]), ("o200k", fert["o2"]))},
    "tcr_llmlingua2_dev": {"n": int(len(ll2)),
                           "corr": round(float(np.corrcoef(ll2.tcr, ll2.tcr_o2)[0, 1]), 3),
                           "mean_cl100k": round(float(ll2.tcr.mean()), 3),
                           "mean_o200k": round(float(ll2.tcr_o2.mean()), 3)},
}

# exam data, frozen selectors ---------------------------------------------------
er = pd.read_csv(E11 / "exam_results.csv")
ef = pd.read_csv(E11 / "exam_features.csv").set_index("prompt_id")
cats = ef.task_category
piv = ev.tables(er)
ids = [p for p in ef.index if p in piv["tcr"].index
       and piv["out_f1_arabert"].loc[p, PRIMARY].notna().all()]
v1 = json.loads((ev.SEL / "apcs_v1.json").read_text())
v2t = json.loads((ev.SEL / "apcs_v2.json").read_text())
vc = json.loads((ev.SEL / "apcs_cost.json").read_text())
pol = {"APCS-v2": pd.Series([ev.walk(v2t["tree"], ef.loc[p]) for p in ids], index=ids),
       "APCS-1.0.0": pd.Series(ev.apply_v1(v1, ef.loc[ids].token_count.to_numpy()), index=ids),
       "APCS-cost": pd.Series([ev.walk(vc["tree"], ef.loc[p]) for p in ids], index=ids),
       "always llmlingua2@0.5": pd.Series("llmlingua2@0.5", index=ids)}


def bb_labels(tau_of):
    tcr, f1 = piv["tcr"], piv["out_f1_arabert"]
    out = []
    for p in ids:
        ok = [c for c in PRIMARY if f1.loc[p, c] >= tau_of(p)]
        out.append(max(ok, key=lambda c: tcr.loc[p, c]) if ok else "noop@1.0")
    return pd.Series(out, index=ids)


def mc(a, b, lab, sel=None):
    sel = ids if sel is None else sel
    ca, cb = (pol[a][sel] == lab[sel]).to_numpy(), (pol[b][sel] == lab[sel]).to_numpy()
    ao, bo, p = mcnemar_exact(ca, cb)
    return {"n": len(sel), "acc_a": round(float(ca.mean()), 4), "acc_b": round(float(cb.mean()), 4),
            "diff": round(float(ca.mean() - cb.mean()), 4), "a_only": ao, "b_only": bo,
            "p": round(float(p), 4)}


lab65 = bb_labels(lambda p: 0.65)

# M ---------------------------------------------------------------------------
R["M_H1_by_category"] = {"all": mc("APCS-v2", "always llmlingua2@0.5", lab65)}
for c in CATS:
    R["M_H1_by_category"][f"without_{c}"] = mc(
        "APCS-v2", "always llmlingua2@0.5", lab65, [p for p in ids if cats[p] != c])
R["M_H1_by_category"]["net_gain_prompts_by_category"] = {
    c: int(((pol["APCS-v2"] == lab65) & (cats[ids] == c)).sum()
           - ((pol["always llmlingua2@0.5"] == lab65) & (cats[ids] == c)).sum()) for c in CATS}

# N ---------------------------------------------------------------------------
cost = piv["cost_usd"]
base = np.array([cost.loc[p, "noop@1.0"] for p in ids])
cc = np.array([cost.loc[p, c] for p, c in pol["APCS-cost"].items()])
dc = cc - base
boot = []
for _ in range(10000):
    i = rng.integers(len(ids), size=len(ids))
    boot.append(cc[i].sum() / base[i].sum() - 1)
perm = np.array([(dc * rng.choice([-1, 1], size=len(dc))).mean() for _ in range(20000)])
R["N_H3_total"] = {"total_change": round(float(cc.sum() / base.sum() - 1), 4),
                   "ci95": [round(float(np.percentile(boot, 2.5)), 4),
                            round(float(np.percentile(boot, 97.5)), 4)],
                   "zero_differences": int((dc == 0).sum()), "n": len(dc),
                   "share_choosing_none": round(float((pol["APCS-cost"] == "noop@1.0").mean()), 4),
                   "signflip_p_one_sided_mean": round(float((perm <= dc.mean()).mean()), 4)}

# O ---------------------------------------------------------------------------
cat_tau = {"qa": 0.768, "summarisation": 0.713, "instruction": 0.613, "creative": 0.636}  # Table 11
R["O_tau_sensitivity"] = {}
for name, fn in [("0.60", lambda p: 0.60), ("0.65", lambda p: 0.65), ("0.70", lambda p: 0.70),
                 ("per_category_0.4th_pct", lambda p: cat_tau[cats[p]])]:
    lab = bb_labels(fn)
    h1 = mc("APCS-v2", "always llmlingua2@0.5", lab)
    h2 = mc("APCS-v2", "APCS-1.0.0", lab)
    order = sorted([("H1", h1["p"]), ("H2", h2["p"])], key=lambda x: x[1])
    holm = {order[0][0]: min(1, 2 * order[0][1])}
    holm[order[1][0]] = max(holm[order[0][0]], min(1, order[1][1]))
    R["O_tau_sensitivity"][name] = {
        "label_share_none": round(float((lab == "noop@1.0").mean()), 4),
        "H1": h1, "H2": h2, "holm_over_H1_H2": {k: round(v, 4) for k, v in holm.items()}}

# P ---------------------------------------------------------------------------
feats = pd.read_csv(E10 / "features_final.csv")
ceil = pd.read_csv(E10 / "ceiling.csv")[["prompt_id", "ceiling_f1_arabert"]]
foc = fr[(fr.origin == "v2") & (fr.split == "dev") & (fr.method == "llmlingua2")
         & (fr.target_rate == 0.5)].merge(feats.drop(columns=["task_category"]), on="prompt_id")
foc = foc.merge(ceil, on="prompt_id")
foc["log_tokens"] = np.log(foc.token_count)
for f in ["log_tokens", "structural_complexity", "fragmentation_ratio", "morphological_density",
          "ceiling_f1_arabert"]:
    foc[f"z_{f}"] = (foc[f] - foc[f].mean()) / foc[f].std()
zs = "z_log_tokens + z_structural_complexity + z_fragmentation_ratio + z_morphological_density"
fits = {k: smf.ols(f, data=foc).fit() for k, f in {
    "full": f"out_f1_arabert ~ {zs} + C(category)",
    "full_noise": f"out_f1_arabert ~ {zs} + z_ceiling_f1_arabert + C(category)",
    "noise_no_cat": f"out_f1_arabert ~ {zs} + z_ceiling_f1_arabert",
    "noise_no_len": "out_f1_arabert ~ z_structural_complexity + z_fragmentation_ratio + "
                    "z_morphological_density + z_ceiling_f1_arabert + C(category)"}.items()}
fn = fits["full_noise"]
R["P_rq2_noise_covariate"] = {
    "n": int(len(foc)), "r2_without_noise": round(float(fits["full"].rsquared), 4),
    "r2_with_noise": round(float(fn.rsquared), 4),
    "unique_r2_category_with_noise": round(float(fn.rsquared - fits["noise_no_cat"].rsquared), 4),
    "unique_r2_length_with_noise": round(float(fn.rsquared - fits["noise_no_len"].rsquared), 4),
    "beta_noise": round(float(fn.params["z_ceiling_f1_arabert"]), 4),
    "p_noise": float(fn.pvalues["z_ceiling_f1_arabert"]),
    "category_effects_vs_creative": {k: {"b": round(float(fn.params[k]), 4), "p": float(fn.pvalues[k])}
                                     for k in fn.params.index if k.startswith("C(category)")},
    "beta_morphology": round(float(fn.params["z_morphological_density"]), 4),
    "p_morphology": float(fn.pvalues["z_morphological_density"]),
    "beta_fragmentation": round(float(fn.params["z_fragmentation_ratio"]), 4),
    "p_fragmentation": float(fn.pvalues["z_fragmentation_ratio"]),
}

# Q ---------------------------------------------------------------------------
dev = fr[(fr.origin == "v2") & (fr.split == "dev")].copy()
dev["cost_usd"] = 0.0
dev["qa_contains"] = np.nan
dpiv = ev.tables(dev)
dids = [p for p in dpiv["tcr"].index if dpiv["out_f1_arabert"].loc[p, PRIMARY].notna().all()]
dlab = []
for p in dids:
    ok = [c for c in PRIMARY if dpiv["out_f1_arabert"].loc[p, c] >= 0.65]
    dlab.append(max(ok, key=lambda c: dpiv["tcr"].loc[p, c]) if ok else "noop@1.0")
dlab = pd.Series(dlab, index=dids)
dcat = dev.drop_duplicates("prompt_id").set_index("prompt_id").category
lookup = {c: dlab[[p for p in dids if dcat[p] == c]].value_counts().idxmax() for c in CATS}
pol["per-category lookup"] = pd.Series([lookup[cats[p]] for p in ids], index=ids)
R["Q_per_category_lookup_exploratory"] = {
    "rule_from_dev": lookup,
    "vs_APCS-v2": mc("APCS-v2", "per-category lookup", lab65),
    "vs_always@0.5": mc("per-category lookup", "always llmlingua2@0.5", lab65)}

# R ---------------------------------------------------------------------------
num = re.compile(r"(\d+|[٠-٩]+|عشر|عشرين|ثلاثين|أربعين|خمسين|مائة|مئة|ألف|مئتي|مائتي|خمسمائة|ثلاثمائة)"
                 r"\s*(?:\S+\s+)?كلمة")
rows = []
for f in ["AraPromptBench_v2.json", "AraPromptBench_exam.json"]:
    d = json.loads((PROJECT / f).read_text(encoding="utf-8"))
    rows += d["prompts"] if isinstance(d, dict) else d
flag = [it for it in rows if LENGTH_MARKERS.search(it["prompt"])]
only_k = [it for it in flag if {m.group(0) for m in LENGTH_MARKERS.finditer(it["prompt"])} <= {"كلمة"}]
no_num = [it for it in only_k if not (num.search(it["prompt"]) or "حدود" in it["prompt"])]
R["R_marker_precision"] = {"prompts": len(rows), "flagged": len(flag), "flagged_only_by_kalima": len(only_k),
                           "kalima_without_number": len(no_num),
                           "est_false_positive_share": round(len(no_num) / len(flag), 3),
                           "examples": [it["id"] for it in no_num[:10]]}

# S ---------------------------------------------------------------------------
qa = [p for p in ids if cats[p] == "qa"]
R["S_apcs_cost_qa"] = {"choices": pol["APCS-cost"][qa].value_counts().to_dict(),
                       "compressed_share": round(float((pol["APCS-cost"][qa] != "noop@1.0").mean()), 2)}

# O2 --------------------------------------------------------------------------
# Fairer tau check: the frozen trees were trained on tau = 0.65 labels. Here the whole
# design is repeated on DEV at each tau (same CV procedure as design_selectors.py), the
# comparator is the best fixed strategy on dev at that tau, and the exam is used once.
import design_selectors as ds  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

dpiv2, DX, dcats = ds.load()
dev_ids = list(DX.index)
EX = ef.loc[ids, ds.FEATURES]


def bb(pv, pids, tau_of):
    out = []
    for p in pids:
        ok = [c for c in PRIMARY if pv["out_f1_arabert"].loc[p, c] >= tau_of(p)]
        out.append(max(ok, key=lambda c: pv["tcr"].loc[p, c]) if ok else "noop@1.0")
    return np.array(out)


R["O2_tau_redesigned_on_dev"] = {}
for name, tf in [("0.60", lambda c: 0.60), ("0.65", lambda c: 0.65), ("0.70", lambda c: 0.70),
                 ("per_category_0.4th_pct", lambda c: cat_tau[c])]:
    y = bb(dpiv2, dev_ids, lambda p: tf(dcats[p]))
    res, ref = ds.cross_validate(DX, y, dpiv2, dcats)
    d, m = ds.pick(res)
    tree = DecisionTreeClassifier(max_depth=d, min_samples_leaf=m, random_state=ds.SEED).fit(DX, y)
    rule = ds.fit_v1(DX.token_count.to_numpy(), y)
    fixed = max(PRIMARY, key=lambda c: np.mean(y == c))
    lab = bb_labels(lambda p: tf(cats[p]))
    pol["tree@tau"] = pd.Series(tree.predict(EX), index=ids)
    pol["fixed@tau"] = pd.Series(fixed, index=ids)
    pol["rule@tau"] = pd.Series(ds.v1_rule(EX.token_count.to_numpy(), **rule), index=ids)
    R["O2_tau_redesigned_on_dev"][name] = {
        "tree": {"depth": d, "leaf": m, "dev_cv_accuracy": round(res[(d, m)]["cv_accuracy"], 4)},
        "best_fixed_on_dev": fixed, "dev_rule": rule,
        "tree_vs_best_fixed": mc("tree@tau", "fixed@tau", lab),
        "tree_vs_rule": mc("tree@tau", "rule@tau", lab)}

R["gap_closed_by_APCS-v2"] = round((0.385 - 0.3275) / (1 - 0.3275), 3)

OUT.mkdir(exist_ok=True)
(OUT / "review_checks.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str),
                                        encoding="utf-8")
print(json.dumps(R, indent=1, ensure_ascii=False, default=str))
