"""Exp 13 (supplementary) — analyses requested in the DA's draft feedback.

All computed from stored results (no new API calls):
  A  cl100k fragmentation of Arabic clitic words vs English equivalents
  B  tokens per Arabic word for candidate LLMLingua scorer tokenisers
  C  repeat-call noise ceiling by task category (dev)
  D  QA: does the gold answer survive compression, and does that predict correctness (exam)
  E  morphological density: spread and collinearity (dev)
  F  cost break-even: how much answer growth cancels a given input saving (dev)
  G  APCS-v2 on exam QA prompts: accuracy by tree leaf
  H  BERTScore sensitivity to answer-length change (dev, LLMLingua-2 @ 0.5)
  I  precision and minimum detectable effect of the 200-prompt test and 400-prompt exam
  J  qualitative cases: LLMLingua-2 vs random deletion (dev)
  K  length-instruction regex: exact pattern and coverage
Output: results/supplementary.json (+ results/cases.json)
"""

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken
from scipy import stats

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
E10 = PROJECT / "experiments" / "10_final_benchmark" / "results"
E11 = PROJECT / "experiments" / "11_fresh_exam" / "results"
OUT = ROOT / "results"
sys.path.insert(0, str(PROJECT / "experiments" / "10_final_benchmark"))
sys.path.insert(0, str(PROJECT / "apcs"))
from qa_metrics import normalise  # noqa: E402
from apcs.features import LENGTH_MARKERS  # noqa: E402
from apcs.selector import tree_inputs, walk_tree  # noqa: E402
from apcs import extract_features  # noqa: E402

ENC = tiktoken.get_encoding("cl100k_base")
P_IN, P_OUT, TAU = 0.20, 1.20, 0.65
R = {}


def pieces(text):
    out = []
    for t in ENC.encode(text):
        b = ENC.decode_single_token_bytes(t)
        try:
            out.append(b.decode("utf-8"))
        except UnicodeDecodeError:
            out.append("<" + b.hex(" ") + ">")
    return out


# A ---------------------------------------------------------------------------
words = {"وبالكتاب": "wa+bi+al+kitab (and with the book)",
         "فسيكتبونها": "fa+sa+yaktubuna+ha (so they will write it)",
         "والمعلمون": "wa+al+muallimun (and the teachers)",
         "بالمدرسة": "bi+al+madrasa (at the school)"}
eng = ["and with the book", "so they will write it", "and the teachers", "at the school"]
R["A_clitic_examples"] = [
    {"arabic": w, "gloss": g, "cl100k_tokens": len(ENC.encode(w)), "pieces": pieces(w),
     "english": e, "english_tokens": len(ENC.encode(e)), "english_words": len(e.split())}
    for (w, g), e in zip(words.items(), eng)]

# B ---------------------------------------------------------------------------
v2 = json.loads((PROJECT / "AraPromptBench_v2.json").read_text(encoding="utf-8"))["prompts"]
dev_prompts = [p["prompt"] for p in v2 if p["split"] == "dev"]
rng = np.random.default_rng(42)
sample = [dev_prompts[i] for i in rng.choice(len(dev_prompts), 200, replace=False)]
nwords = sum(len(t.split()) for t in sample)
fert = {"cl100k_base (LLM under test, reference)": sum(len(ENC.encode(t)) for t in sample) / nwords}
try:
    from transformers import AutoTokenizer
    for mid in ["openai-community/gpt2", "NousResearch/Llama-2-7b-hf", "Qwen/Qwen2.5-0.5B",
                "Qwen/Qwen3-0.6B", "NousResearch/Meta-Llama-3.1-8B", "aubmindlab/aragpt2-base",
                "inceptionai/jais-family-590m", "FacebookAI/xlm-roberta-large",
                "google-bert/bert-base-multilingual-cased"]:
        tok = AutoTokenizer.from_pretrained(mid)
        fert[mid] = sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in sample) / nwords
except Exception as e:  # tokenizers are cached locally; report if not
    fert["error"] = str(e)[:200]
R["B_tokens_per_word"] = {k: round(v, 2) if isinstance(v, float) else v for k, v in fert.items()}

# C ---------------------------------------------------------------------------
ceil = pd.read_csv(E10 / "ceiling.csv")
dev_c = ceil[(ceil.origin == "v2") & (ceil.split == "dev")]
R["C_ceiling_by_category"] = {
    c: {"n": int(len(g)), "identical_pct": round(100 * g.identical.mean(), 1),
        "median_f1": round(float(g.ceiling_f1_arabert.median()), 3),
        "pct_below_0.85": round(100 * float((g.ceiling_f1_arabert < .85).mean()), 1),
        "q0.004": round(float(g.ceiling_f1_arabert.quantile(.004)), 3)}
    for c, g in dev_c.groupby("category")}
R["C_tau"] = {"q0.004_all_dev": round(float(dev_c.ceiling_f1_arabert.quantile(.004)), 4),
              "n_dev": int(len(dev_c))}

# D ---------------------------------------------------------------------------
exam = {p["id"]: p for p in json.loads((PROJECT / "AraPromptBench_exam.json")
                                       .read_text(encoding="utf-8"))["prompts"]}
er = pd.read_csv(E11 / "exam_results.csv")
er["cand"] = er.method + "@" + er.target_rate.astype(str)
qa = er[er.category == "qa"].copy()


def golds(pid):
    g = exam[pid].get("gold_answers")
    return ast.literal_eval(g) if isinstance(g, str) else (g or [])


qa["gold_kept"] = [int(any(f" {normalise(g)} " in f" {normalise(str(c))} " for g in golds(p) if normalise(g)))
                   for p, c in zip(qa.prompt_id, qa.compressed)]
rows = {}
for cand, g in qa.groupby("cand"):
    kept, lost = g[g.gold_kept == 1], g[g.gold_kept == 0]
    rows[cand] = {"gold_kept_pct": round(100 * g.gold_kept.mean(), 1),
                  "correct_if_kept": round(100 * kept.qa_contains.mean(), 1) if len(kept) else None,
                  "correct_if_lost": round(100 * lost.qa_contains.mean(), 1) if len(lost) else None,
                  "correct_all": round(100 * g.qa_contains.mean(), 1)}
R["D_qa_gold_retention"] = rows
comp = qa[qa.cand != "noop@1.0"]
R["D_overall"] = {"correct_if_kept": round(100 * comp[comp.gold_kept == 1].qa_contains.mean(), 1),
                  "correct_if_lost": round(100 * comp[comp.gold_kept == 0].qa_contains.mean(), 1),
                  "n_variants": int(len(comp))}

# E ---------------------------------------------------------------------------
feat = pd.read_csv(E10 / "features_final.csv").set_index("prompt_id")
devids = [p["id"] for p in v2 if p["split"] == "dev"]
f = feat.loc[devids]
md = f.morphological_density
R["E_morphology"] = {
    "mean": round(float(md.mean()), 3), "sd": round(float(md.std()), 3),
    "cv_pct": round(100 * float(md.std() / md.mean()), 1),
    "p5_p95": [round(float(md.quantile(.05)), 3), round(float(md.quantile(.95)), 3)],
    "spearman_with_fragmentation": round(float(stats.spearmanr(md, f.fragmentation_ratio)[0]), 3),
    "spearman_with_log_tokens": round(float(stats.spearmanr(md, np.log(f.token_count))[0]), 3)}
import statsmodels.api as sm  # noqa: E402
X = pd.DataFrame({"log_len": np.log(f.token_count), "frag": f.fragmentation_ratio,
                  "struct": f.structural_complexity, "morph": md})
X = sm.add_constant((X - X.mean()) / X.std())
R["E_vif"] = {c: round(float(1 / (1 - sm.OLS(X[c], X.drop(columns=c)).fit().rsquared)), 2)
              for c in ["log_len", "frag", "struct", "morph"]}

# F ---------------------------------------------------------------------------
fr = pd.read_csv(E10 / "final_results.csv")
fr = fr[(fr.origin == "v2") & (fr.split == "dev")].copy()
fr["cand"] = fr.method + "@" + fr.target_rate.astype(str)
base = fr[fr.cand == "noop@1.0"]
n_in, n_out = base.resp_prompt_tokens.sum(), base.resp_completion_tokens.sum()
F = {"mean_input_tokens_uncompressed": round(float(base.resp_prompt_tokens.mean()), 1),
     "mean_output_tokens_uncompressed": round(float(base.resp_completion_tokens.mean()), 1),
     "input_output_ratio": round(float(n_in / n_out), 3), "policies": {}}
for cand in ["llmlingua2@0.7", "llmlingua2@0.5", "llmlingua2@0.3"]:
    g = fr[fr.cand == cand]
    s = 1 - g.resp_prompt_tokens.sum() / n_in
    growth = g.resp_completion_tokens.sum() / n_out - 1
    breakeven = (P_IN / P_OUT) * s * (n_in / n_out)
    F["policies"][cand] = {"input_saving_pct": round(100 * s, 1),
                           "breakeven_output_growth_pct": round(100 * breakeven, 1),
                           "observed_output_growth_pct": round(100 * growth, 1)}
R["F_cost_breakeven"] = F

# G ---------------------------------------------------------------------------
sel = json.loads((PROJECT / "apcs" / "apcs" / "selectors" / "apcs_v2.json").read_text())
ef = pd.read_csv(E11 / "exam_features.csv").set_index("prompt_id")
pv = er.pivot_table(index="prompt_id", columns="cand", values="tcr")
pf = er.pivot_table(index="prompt_id", columns="cand", values="out_f1_arabert")
PRIMARY = ["noop@1.0", "llmlingua2@0.7", "llmlingua2@0.5", "llmlingua2@0.3"]
leaf_rows = []
for pid in [p for p in pv.index if exam[p]["category"] == "qa"]:
    ok = [c for c in PRIMARY if pf.loc[pid, c] >= TAU]
    label = max(ok, key=lambda c: pv.loc[pid, c]) if ok else "noop@1.0"
    leaf, path = walk_tree(sel["tree"], ef.loc[pid].to_dict())
    leaf_rows.append({"pid": pid, "label": label, "pred": leaf, "path": " & ".join(path[1:]),
                      "tokens": int(ef.loc[pid, "token_count"])})
L = pd.DataFrame(leaf_rows)
R["G_qa_leaves"] = {
    path: {"n": int(len(g)), "pred": g.pred.iloc[0],
           "v2_correct_pct": round(100 * float((g.pred == g.label).mean()), 1),
           "fixed05_correct_pct": round(100 * float((g.label == "llmlingua2@0.5").mean()), 1),
           "label_mix": g.label.value_counts().to_dict()}
    for path, g in L.groupby("path")}

# H ---------------------------------------------------------------------------
pf_dev = fr.pivot_table(index="prompt_id", columns="cand", values="out_f1_arabert")
pc_dev = fr.pivot_table(index="prompt_id", columns="cand", values="resp_completion_tokens")
ratio = np.log(pc_dev["llmlingua2@0.5"] / pc_dev["noop@1.0"].clip(lower=1))
rho, p = stats.spearmanr(ratio.abs(), pf_dev["llmlingua2@0.5"], nan_policy="omit")
grow = ratio > np.log(1.5)
R["H_length_sensitivity"] = {
    "spearman_abs_log_length_ratio_vs_f1": round(float(rho), 3), "p": float(p),
    "mean_f1_answer_grew_over_50pct": round(float(pf_dev["llmlingua2@0.5"][grow].mean()), 3),
    "mean_f1_other": round(float(pf_dev["llmlingua2@0.5"][~grow].mean()), 3),
    "share_grew_over_50pct": round(100 * float(grow.mean()), 1)}

# I ---------------------------------------------------------------------------
apcs_final = json.loads((E10 / "apcs_final.json").read_text())["primary_arabert"]
t05 = apcs_final["apcs_vs_baselines"]["always llmlingua2@0.5"]
pd_test = (t05["apcs_only_correct"] + t05["baseline_only_correct"]) / 200
pd_exam = (62 + 39) / 400
z = stats.norm.ppf(.975) + stats.norm.ppf(.80)
R["I_power"] = {
    "test_n200": {"accuracy_ci_halfwidth_pts": round(100 * 1.96 * np.sqrt(.3 * .7 / 200), 1),
                  "discordant_share": round(pd_test, 3),
                  "mde_80pct_power_pts": round(100 * z * np.sqrt(pd_test / 200), 1)},
    "exam_n400": {"accuracy_ci_halfwidth_pts": round(100 * 1.96 * np.sqrt(.35 * .65 / 400), 1),
                  "discordant_share": round(pd_exam, 3),
                  "mde_80pct_power_pts": round(100 * z * np.sqrt(pd_exam / 400), 1)}}

# J ---------------------------------------------------------------------------
pt = fr.pivot_table(index="prompt_id", columns="cand", values="out_f1_arabert")
cands = pt[(pt["llmlingua2@0.5"] >= .80) & (pt["random_deletion@0.5"] <= .55)].index
info = fr.drop_duplicates("prompt_id").set_index("prompt_id")
picked = []
for cat in ["instruction", "summarisation", "qa"]:
    ids = [p for p in cands if info.loc[p, "category"] == cat and info.loc[p, "orig_tokens"] <= 120]
    if ids:
        picked.append(ids[0])
cases = []
for pid in picked:
    g = fr[fr.prompt_id == pid].set_index("cand")
    cases.append({"id": pid, "category": info.loc[pid, "category"],
                  "original": g.loc["noop@1.0", "compressed"],
                  "llmlingua2_0.5": g.loc["llmlingua2@0.5", "compressed"],
                  "random_0.5": g.loc["random_deletion@0.5", "compressed"],
                  "answer_original": str(g.loc["noop@1.0", "response"])[:400],
                  "answer_llmlingua2": str(g.loc["llmlingua2@0.5", "response"])[:400],
                  "answer_random": str(g.loc["random_deletion@0.5", "response"])[:400],
                  "f1_llmlingua2": round(float(g.loc["llmlingua2@0.5", "out_f1_arabert"]), 3),
                  "f1_random": round(float(g.loc["random_deletion@0.5", "out_f1_arabert"]), 3)})

# K ---------------------------------------------------------------------------
allp = [p["prompt"] for p in v2] + [p["prompt"] for p in exam.values()]
markers = LENGTH_MARKERS.pattern.split("|")
R["K_regex"] = {"pattern": LENGTH_MARKERS.pattern,
                "matches_per_marker": {m: sum(m in t for t in allp) for m in markers},
                "prompts_matched": sum(bool(LENGTH_MARKERS.search(t)) for t in allp),
                "n_prompts": len(allp),
                "unmatched_variants_present": {v: sum(v in t and not LENGTH_MARKERS.search(t) for t in allp)
                                               for v in ["في حدود", "لا تتجاوز", "باختصار", "بشكل مختصر"]}}

OUT.mkdir(exist_ok=True)
(OUT / "supplementary.json").write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
(OUT / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(R, ensure_ascii=False, indent=1))
print("cases:", [c["id"] for c in cases])
