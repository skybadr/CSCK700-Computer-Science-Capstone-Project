"""Pre-writing audit: recompute every headline number from raw data.

Independent re-implementation (does not import any experiment code) of the
numbers the dissertation will cite. Each check prints PASS / FAIL / WARN and
the audit exits non-zero on any FAIL. Run from the project root:

    .venv/Scripts/python experiments/AUDIT/verify_results.py
"""

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken
from scipy import stats

PROJECT = Path(__file__).resolve().parents[2]
EXP = PROJECT / "experiments"
E10 = EXP / "10_final_benchmark" / "results"
E11 = EXP / "11_fresh_exam" / "results"
P_IN, P_OUT = 0.20, 1.20
TAU = 0.65
PRIMARY = ["noop@1.0", "llmlingua2@0.7", "llmlingua2@0.5", "llmlingua2@0.3"]

results = []


def check(name, ok, detail="", warn=False):
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    results.append((status, name, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


def close(a, b, tol):
    return abs(a - b) <= tol


def norm(t):
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[ً-ْـ]", "", t)          # harakat, tatweel
    return re.sub(r"\s+", " ", t).strip()


def shingles(t, n=12):
    w = norm(t).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def load_prompts(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["prompts"] if isinstance(d, dict) else d


def text_of(r):
    return r.get("prompt") or r.get("text") or r.get("prompt_text")


# ----------------------------------------------------------------- datasets
def audit_datasets():
    print("\n== A. Datasets")
    pilot = load_prompts(PROJECT / "AraPromptBench_dataset.json")
    v2 = load_prompts(PROJECT / "AraPromptBench_v2.json")
    exam = load_prompts(PROJECT / "AraPromptBench_exam.json")
    probe = load_prompts(EXP / "09_synthetic_sensitivity" / "probe_pool.json")

    check("pilot has 500 prompts", len(pilot) == 500, str(len(pilot)))
    check("v2 has 1,000 prompts", len(v2) == 1000, str(len(v2)))
    check("exam has 400 prompts", len(exam) == 400, str(len(exam)))
    check("probe pool has 456 prompts", len(probe) == 456, str(len(probe)))

    for name, ds in [("pilot", pilot), ("v2", v2), ("exam", exam), ("probe", probe)]:
        ids = [r["id"] for r in ds]
        texts = [norm(text_of(r)) for r in ds]
        check(f"{name}: unique ids", len(set(ids)) == len(ids))
        check(f"{name}: no empty prompts", all(t for t in texts))
        dup = len(texts) - len(set(texts))
        check(f"{name}: no duplicate prompt texts", dup == 0, f"{dup} duplicates")

    cat = Counter(r["category"] for r in v2)
    check("v2 categories 350/250/250/150",
          dict(cat) == {"instruction": 350, "summarisation": 250, "qa": 250, "creative": 150},
          str(dict(cat)))
    split = Counter(r["split"] for r in v2)
    check("v2 split 800 dev / 200 test", split == Counter(dev=800, test=200), str(dict(split)))
    by = Counter((r["category"], r["split"]) for r in v2)
    strat = all(close(by[(c, "test")] / n, 0.2, 0.01) for c, n in cat.items())
    check("v2 test split stratified 20% per category", strat,
          str({c: by[(c, 'test')] for c in cat}))
    synth = sum(1 for r in v2 if "synthetic" in str(r.get("source", "")).lower()
                or "claude" in str(r.get("source", "")).lower())
    check("v2 has 150 synthetic / 850 sourced", synth == 150, f"{synth} synthetic")

    ecat = Counter(r["category"] for r in exam)
    check("exam categories 140/100/100/60",
          dict(ecat) == {"instruction": 140, "summarisation": 100, "qa": 100, "creative": 60},
          str(dict(ecat)))

    # contamination: exact and 12-word partial overlap, computed independently
    # shared instruction templates (spans in >= 5 prompts of a set) are
    # identical by design; contamination means a shared passage
    def content_shingles(ds):
        cnt = Counter(s for r in ds for s in shingles(text_of(r)))
        templ = {s for s, n in cnt.items() if n >= 5}
        return set(cnt) - templ, templ
    sets = {"pilot": pilot, "v2": v2, "probe": probe}
    exam_txt = {r["id"]: norm(text_of(r)) for r in exam}
    templ_all = set().union(*(content_shingles(d)[1] for d in [pilot, v2, probe, exam]))
    exam_sh = {r["id"]: shingles(text_of(r)) - templ_all for r in exam}
    for name, ds in sets.items():
        txt = {norm(text_of(r)) for r in ds}
        sh = content_shingles(ds)[0]
        exact = sum(t in txt for t in exam_txt.values())
        part = sum(bool(s & sh) for s in exam_sh.values())
        check(f"exam vs {name}: zero exact overlap", exact == 0, str(exact))
        check(f"exam vs {name}: zero 12-word passage overlap (templates excluded)", part == 0, str(part))
    v2_txt = {norm(text_of(r)) for r in v2}
    v2_sh = set().union(*(shingles(text_of(r)) for r in v2 if r["split"] == "test"))
    pil_exact = sum(norm(text_of(r)) in v2_txt for r in pilot)
    check("v2 vs pilot: zero exact overlap", pil_exact == 0, str(pil_exact))
    pil_part = sum(bool(shingles(text_of(r)) & v2_sh) for r in pilot)
    check("v2 TEST vs pilot: zero 12-word overlap", pil_part == 0,
          f"{pil_part} pilot prompts share a 12-word span with a test prompt", warn=True)
    probe_part = sum(bool(shingles(text_of(r)) & v2_sh) for r in probe)
    check("v2 TEST vs probe: zero 12-word overlap", probe_part == 0,
          f"{probe_part} probe prompts share a 12-word span with a test prompt", warn=True)
    return v2, exam


# ----------------------------------------------------------------- Exp 10 raw
def audit_exp10_raw(v2):
    print("\n== B. Exp 10 raw data integrity")
    resp = [json.loads(l) for l in open(E10 / "responses.jsonl", encoding="utf-8")]
    keys = [r["key"] for r in resp]
    check("13,438 API responses", len(resp) == 13438, str(len(resp)))
    check("response keys unique", len(set(keys)) == len(keys))
    models = Counter(r["model"] for r in resp)
    check("single model snapshot (gpt-5.6-luna)", list(models) == ["gpt-5.6-luna"], str(dict(models)))
    fin = Counter(r["finish_reason"] for r in resp)
    check("79 truncated answers", fin.get("length", 0) == 79, str(dict(fin)))
    empty = sum(1 for r in resp if not str(r["response"]).strip())
    check("exactly 1 empty answer", empty == 1, str(empty))
    pt = sum(int(r["prompt_tokens"]) for r in resp)
    ct = sum(int(r["completion_tokens"]) for r in resp)
    cost = (pt * P_IN + ct * P_OUT) / 1e6
    check("total cost $2.49", close(cost, 2.49, 0.006), f"${cost:.4f} ({pt:,} in / {ct:,} out)")

    df = pd.read_csv(E10 / "final_results.csv")
    df["cand"] = df.method + "@" + df.target_rate.astype(str)
    dups = df.duplicated(["prompt_id", "cand"]).sum()
    check("final_results: one row per prompt × candidate", dups == 0, f"{dups} duplicates")
    v2ids = {r["id"]: r for r in v2}
    sub = df[df.origin == "v2"]
    check("final_results covers all 1,000 v2 prompts", sub.prompt_id.nunique() == 1000,
          str(sub.prompt_id.nunique()))
    mism = sum(v2ids[p]["split"] != s for p, s in zip(sub.prompt_id, sub.split))
    check("split column matches frozen dataset", mism == 0, f"{mism} mismatches")
    mism = sum(v2ids[p]["category"] != c for p, c in zip(sub.prompt_id, sub.category))
    check("category column matches frozen dataset", mism == 0, f"{mism} mismatches")

    # rejoin answers by call_key and compare text
    rk = {r["key"]: r for r in resp}
    has = df.call_key.notna()
    missing = (~df.loc[has, "call_key"].isin(rk)).sum()
    check("every call_key resolves to a stored response", missing == 0, f"{missing} missing")
    bad = sum(str(rk[k]["response"]) != str(t if pd.notna(t) else "")
              for k, t in zip(df.loc[has, "call_key"], df.loc[has, "response"]))
    check("stored response text matches results table", bad == 0, f"{bad} mismatches")
    badtok = sum(int(rk[k]["completion_tokens"]) != int(c)
                 for k, c in zip(df.loc[has, "call_key"], df.loc[has, "resp_completion_tokens"]))
    check("completion-token counts match results table", badtok == 0, f"{badtok} mismatches")

    scored = df[df.out_f1_arabert.notna()]
    check("all output F1 in [0, 1]",
          scored.out_f1_arabert.between(0, 1.0000001).all() and
          scored.out_f1_mbert.between(0, 1.0000001).all())
    unscored = df[df.call_key.notna() & df.out_f1_arabert.isna()]
    check("every answered row is scored", len(unscored) == 0, f"{len(unscored)} unscored")

    # TCR recomputed with cl100k
    enc = tiktoken.get_encoding("cl100k_base")
    samp = df[df.compressed.notna()].sample(400, random_state=1)
    tok_ok = sum(len(enc.encode(c)) == t for c, t in zip(samp.compressed, samp.comp_tokens))
    check("compressed token counts reproduce with cl100k (sample 400)", tok_ok == 400, f"{tok_ok}/400")
    tcr = 1 - df.comp_tokens / df.orig_tokens
    check("TCR = 1 − comp/orig", np.allclose(tcr, df.tcr, atol=1e-3, equal_nan=True))
    return df


# ----------------------------------------------------------------- Exp 10 findings
def wilcoxon_diff(a, b):
    d = a - b
    return d.mean(), stats.wilcoxon(a, b).pvalue


def audit_exp10_findings(df):
    print("\n== C. Exp 10 headline numbers (recomputed)")
    ceil = pd.read_csv(E10 / "ceiling.csv")
    dev_c = ceil[(ceil.origin == "v2") & (ceil.split == "dev")]
    ceil = dev_c                                   # findings report dev
    ident = ceil.identical.mean()
    check("ceiling: identical repeats 29.6%", close(ident, 0.296, 0.0015), f"{ident:.4f}")
    med = ceil.ceiling_f1_arabert.median()
    below = (ceil.ceiling_f1_arabert < 0.85).mean()
    check("ceiling: median repeat F1 0.930", close(med, 0.930, 0.0015), f"{med:.4f}")
    check("ceiling: 31.8% of identical pairs < 0.85", close(below, 0.318, 0.0015), f"{below:.4f}")
    q = dev_c.ceiling_f1_arabert.quantile(0.004)
    check("τ: 0.4th percentile of dev ceiling ≈ 0.64 → 0.65", close(q, 0.6396, 0.01), f"{q:.4f}")

    dev = df[(df.origin == "v2") & (df.split == "dev")]
    piv = dev.pivot_table(index="prompt_id", columns="cand", values="out_f1_arabert")
    pivp = dev.pivot_table(index="prompt_id", columns="cand", values="f1_arabert")
    exp = {0.3: (0.040, 0.026 * 0 - 0.014), 0.5: (0.038, -0.026), 0.7: (0.050, -0.028)}
    for r, (out_d, pr_d) in exp.items():
        a, b = piv[f"llmlingua2@{r}"], piv[f"random_deletion@{r}"]
        d, p = wilcoxon_diff(a, b)
        check(f"RQ1 out-F1 LL2−random @{r} = {out_d:+.3f}", close(d, out_d, 0.0015) and p < 1e-15,
              f"{d:+.4f}, p={p:.1e}")
        a, b = pivp[f"llmlingua2@{r}"], pivp[f"random_deletion@{r}"]
        d, p = wilcoxon_diff(a, b)
        check(f"ranking flip: prompt-level LL2−random @{r} ≈ {pr_d:+.3f}",
              close(d, pr_d, 0.0015) and d < 0, f"{d:+.4f}, p={p:.1e}")

    # category ceiling-normalised @0.5
    cmed = dev_c.groupby("category").ceiling_f1_arabert.median()
    sub = dev[dev.cand == "llmlingua2@0.5"]
    for c, e in [("creative", 83.7), ("summarisation", 76.7), ("instruction", 75.4), ("qa", 70.6)]:
        v = 100 * sub[sub.category == c].out_f1_arabert.mean() / cmed[c]
        check(f"ceiling-normalised {c} {e}%", close(v, e, 0.15), f"{v:.1f}%")

    # cost table F5 (dev)
    dev = dev.assign(cost=(dev.resp_prompt_tokens * P_IN + dev.resp_completion_tokens * P_OUT) / 1e6)
    cp = dev.pivot_table(index="prompt_id", columns="cand", values="cost")
    ip = dev.pivot_table(index="prompt_id", columns="cand", values="resp_prompt_tokens")
    base = cp["noop@1.0"].sum()
    for cand, e_in, e_tot in [("llmlingua2@0.7", 29.7, 4.6), ("llmlingua2@0.5", 49.1, 10.9),
                              ("llmlingua2@0.3", 67.7, 24.3), ("random_deletion@0.3", 66.1, -27.0),
                              ("llmlingua_qwen@0.3", 23.9, -3.0)]:
        tot = 100 * (cp[cand].sum() / base - 1)
        ins = 100 * (1 - ip[cand].sum() / ip["noop@1.0"].sum())
        check(f"cost {cand}: input −{e_in}% / total {e_tot:+}%",
              close(tot, e_tot, 0.15) and close(ins, e_in, 0.5), f"input −{ins:.1f}%, total {tot:+.1f}%")

    # APCS test accuracy (independent label + rule implementation)
    test = df[(df.origin == "v2") & (df.split == "test")]
    tcr = test.pivot_table(index="prompt_id", columns="cand", values="tcr")
    f1 = test.pivot_table(index="prompt_id", columns="cand", values="out_f1_arabert")
    tok = test.groupby("prompt_id").orig_tokens.first()
    lab = {}
    for p in f1.index:
        ok = [c for c in PRIMARY if f1.loc[p, c] >= TAU]
        lab[p] = max(ok, key=lambda c: tcr.loc[p, c]) if ok else "noop@1.0"
    lab = pd.Series(lab)
    dist = lab.value_counts(normalize=True)
    check("test label mix 29/28.5/22/20.5%",
          close(dist["llmlingua2@0.3"], .29, .001) and close(dist["noop@1.0"], .22, .001),
          str(dist.round(3).to_dict()))
    pred = pd.Series(np.where(tok < 80, "noop@1.0",
                              np.where(tok >= 250, "llmlingua2@0.3", "llmlingua2@0.5")),
                     index=tok.index)
    acc = (pred == lab.loc[pred.index]).mean()
    check("APCS 1.0.0 test accuracy 31.0%", close(acc, 0.31, 0.001), f"{acc:.3f}")
    for c, e in [("llmlingua2@0.5", .285), ("llmlingua2@0.3", .29), ("noop@1.0", .22),
                 ("llmlingua2@0.7", .205)]:
        a = (lab == c).mean()
        check(f"always {c} test accuracy {e:.1%}", close(a, e, 0.001), f"{a:.3f}")
    # McNemar APCS vs never
    a_ok, b_ok = pred == lab, lab == "noop@1.0"
    n01, n10 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue
    check("APCS vs never compress McNemar p = 0.038", close(p, 0.038, 0.002), f"p={p:.3f}")

    # RQ2 OLS unique R²
    import statsmodels.formula.api as smf
    feat = pd.read_csv(E10 / "features_final.csv").set_index("prompt_id")
    d5 = dev[dev.cand == "llmlingua2@0.5"].set_index("prompt_id").join(feat)
    d5 = d5.assign(log_len=np.log(d5.token_count))
    z = lambda s: (s - s.mean()) / s.std()
    for c in ["log_len", "fragmentation_ratio", "structural_complexity", "morphological_density"]:
        d5[c + "_z"] = z(d5[c])
    num = "log_len_z + fragmentation_ratio_z + structural_complexity_z + morphological_density_z"
    full = smf.ols(f"out_f1_arabert ~ {num} + C(category)", d5).fit()
    no_cat = smf.ols(f"out_f1_arabert ~ {num}", d5).fit()
    no_len = smf.ols("out_f1_arabert ~ fragmentation_ratio_z + structural_complexity_z + "
                     "morphological_density_z + C(category)", d5).fit()
    check("RQ2 OLS R² 0.148", close(full.rsquared, 0.148, 0.002), f"{full.rsquared:.3f}")
    uc, ul = full.rsquared - no_cat.rsquared, full.rsquared - no_len.rsquared
    check("RQ2 unique R²: category 6.5%, length 3.4%",
          close(uc, .065, .002) and close(ul, .034, .002), f"category {uc:.3f}, length {ul:.3f}")
    pm = full.pvalues["morphological_density_z"]
    check("RQ2 morphology n.s. (p ≈ 0.65)", pm > 0.05, f"β={full.params['morphological_density_z']:+.4f}, p={pm:.2f}")


# ----------------------------------------------------------------- Exp 11
def walk(node, row):
    while "leaf" not in node:
        node = node["le"] if row[node["feature"]] <= node["threshold"] else node["gt"]
    return node["leaf"]


def audit_exp11():
    print("\n== D. Exp 11 fresh exam (recomputed)")
    resp = [json.loads(l) for l in open(E11 / "responses_exam.jsonl", encoding="utf-8")]
    check("3,828 exam API responses", len(resp) == 3828, str(len(resp)))
    check("single model snapshot", {r["model"] for r in resp} == {"gpt-5.6-luna"})
    cost = sum(int(r["prompt_tokens"]) * P_IN + int(r["completion_tokens"]) * P_OUT for r in resp) / 1e6
    check("exam cost $0.61", close(cost, 0.61, 0.006), f"${cost:.4f}")

    df = pd.read_csv(E11 / "exam_results.csv")
    df["cand"] = df.method + "@" + df.target_rate.astype(str)
    check("exam_results: 400 × 13 rows, no duplicates",
          len(df) == 5200 and not df.duplicated(["prompt_id", "cand"]).any())
    rc = (df.resp_prompt_tokens * P_IN + df.resp_completion_tokens * P_OUT) / 1e6
    check("cost_usd = in×0.20 + out×1.20 per M", np.allclose(rc, df.cost_usd, atol=1e-10))
    check("every exam row scored", df.out_f1_arabert.notna().all())

    piv = {v: df.pivot_table(index="prompt_id", columns="cand", values=v)
           for v in ["tcr", "out_f1_arabert", "cost_usd", "qa_contains"]}
    ids = piv["tcr"].index
    lab = {}
    for p in ids:
        ok = [c for c in PRIMARY if piv["out_f1_arabert"].loc[p, c] >= TAU]
        lab[p] = max(ok, key=lambda c: piv["tcr"].loc[p, c]) if ok else "noop@1.0"
    lab = pd.Series(lab)

    feat = pd.read_csv(E11 / "exam_features.csv").set_index("prompt_id").loc[ids]
    sel = {n: json.loads((E11 / "selectors" / f"{n}.json").read_text()) for n in
           ["apcs_v1", "apcs_v2", "apcs_cost"]}
    t = sel["apcs_v1"]["thresholds"]
    tok = feat.token_count
    pol = {"APCS-1.0.0": pd.Series(np.where(tok < t["T1"], "noop@1.0", np.where(
        tok >= t["T2"], f"llmlingua2@{t['r_long']}", f"llmlingua2@{t['r_mid']}")), index=ids)}
    for n, k in [("APCS-v2", "apcs_v2"), ("APCS-cost", "apcs_cost")]:
        tree = sel[k].get("tree", sel[k])
        pol[n] = pd.Series([walk(tree, feat.loc[p]) for p in ids], index=ids)
    pol["always LL2@0.5"] = pd.Series("llmlingua2@0.5", index=ids)
    pol["never"] = pd.Series("noop@1.0", index=ids)

    def stat(ch):
        pick = lambda v: np.array([piv[v].loc[p, ch[p]] for p in ids])
        return dict(acc=float((ch == lab).mean()),
                    cost=100 * (pick("cost_usd").sum() / piv["cost_usd"]["noop@1.0"].sum() - 1),
                    viol=float((pick("out_f1_arabert") < TAU).mean()))
    s = {n: stat(c) for n, c in pol.items()}
    for n, acc, cst, viol in [("APCS-v2", .385, 15.9, .3425), ("APCS-1.0.0", .3125, 19.7, .345),
                              ("APCS-cost", .195, -1.0, .0925), ("always LL2@0.5", .3275, 12.1, .3975)]:
        r = s[n]
        check(f"exam {n}: acc {acc:.1%}, cost {cst:+}%, below τ {viol:.1%}",
              close(r["acc"], acc, .001) and close(r["cost"], cst, .1) and close(r["viol"], viol, .001),
              f"acc {r['acc']:.4f}, cost {r['cost']:+.2f}%, viol {r['viol']:.4f}")

    def mcn(a, b):
        a_ok, b_ok = pol[a] == lab, pol[b] == lab
        n01, n10 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
        # implementation uses the two-sided exact test (conservative)
        return n01, n10, stats.binomtest(n01, n01 + n10, 0.5).pvalue
    h1 = mcn("APCS-v2", "always LL2@0.5")
    h2 = mcn("APCS-v2", "APCS-1.0.0")
    cv = np.array([piv["cost_usd"].loc[p, pol["APCS-cost"][p]] for p in ids])
    cn = piv["cost_usd"]["noop@1.0"].to_numpy()
    h3p = stats.wilcoxon(cv, cn, alternative="less").pvalue
    ps = np.array([h1[2], h2[2], h3p])
    order = np.argsort(ps)
    holm = np.empty(3)
    run = 0
    for i, j in enumerate(order):
        run = max(run, min(1, (3 - i) * ps[j]))
        holm[j] = run
    check("H1 discordant 62/39, Holm p 0.028", h1[:2] == (62, 39) and close(holm[0], .028, .001),
          f"{h1[:2]}, p_holm={holm[0]:.4f}")
    check("H2 discordant 62/33, Holm p 0.008", h2[:2] == (62, 33) and close(holm[1], .0077, .001),
          f"{h2[:2]}, p_holm={holm[1]:.4f}")
    check("H3 Holm p 0.001", close(holm[2], .0014, .0005), f"p_holm={holm[2]:.4f}")

    qa = [p for p in ids if feat.loc[p, "task_category"] == "qa"]
    qn = piv["qa_contains"].loc[qa, "noop@1.0"].mean()
    check("QA correctness uncompressed 57%", close(qn, .57, .001), f"{qn:.3f}")

    li = feat.index[feat.has_length_instruction.astype(bool)]
    check("180 exam prompts carry a length instruction", len(li) == 180, str(len(li)))
    base = piv["cost_usd"].loc[li, "noop@1.0"].sum()
    for r, e_plain, e_prot in [(0.7, 7.2, 0.8), (0.5, 17.6, -5.3), (0.3, 47.6, -7.9)]:
        pl = 100 * (piv["cost_usd"].loc[li, f"llmlingua2@{r}"].sum() / base - 1)
        pr = 100 * (piv["cost_usd"].loc[li, f"llmlingua2_protected@{r}"].sum() / base - 1)
        check(f"protected @{r}: cost {e_plain:+}% → {e_prot:+}%",
              close(pl, e_plain, .1) and close(pr, e_prot, .1), f"{pl:+.1f}% → {pr:+.1f}%")


# ----------------------------------------------------------------- Exp 09 + pilot
def audit_supporting():
    print("\n== E. Exp 09 and pilot experiments (stored outputs vs findings)")
    sw = pd.read_csv(EXP / "09_synthetic_sensitivity" / "results" / "sweep_09a_summary.csv")
    raw = pd.read_csv(EXP / "09_synthetic_sensitivity" / "results" / "sweep_09a_raw.csv")
    check("09a: 200 repeats per share", (raw.groupby("share").size() == 200).all(),
          str(raw.groupby("share").size().to_dict()))
    check("09a: rates never change (0.5 / 0.3)",
          (sw.r_mid_mode == 0.5).all() and (sw.r_long_mode == 0.3).all())
    check("09a: accuracy 0.380 / 0.377 / 0.374", np.allclose(sw.acc_mean, [.380, .377, .374], atol=.0005))
    b = json.loads((EXP / "09_synthetic_sensitivity" / "results" / "analysis_09b.json")
                   .read_text(encoding="utf-8"))["per_category_rate"]
    excl = sum(not (v["ci95_band_adjusted"][0] <= 0 <= v["ci95_band_adjusted"][1]) for v in b.values())
    check("09b: 8 of 9 intervals exclude zero", excl == 8 and len(b) == 9, f"{excl}/{len(b)}")

    u = json.loads((EXP / "04_output_eval" / "results" / "api_usage.json").read_text())
    check("Exp 04: 4,173 calls, $0.656", u["unique_calls"] == 4173 and u["cost_usd"] == 0.656)
    c7 = pd.read_csv(EXP / "07_protocol_checks" / "results" / "ceiling.csv")
    ident, below = c7.identical_response.mean(), (c7.ceiling_f1_arabert < .85).mean()
    check("Exp 07: 13% identical, 31% below 0.85", close(ident, .13, .006) and close(below, .312, .006),
          f"{ident:.3f}, {below:.3f}")
    cal = json.loads((EXP / "06_apcs_calibration" / "results" / "calibration.json").read_text())
    check("Exp 06: pilot APCS 41.2%", cal["policy_comparison"]["APCS (global rule)"]["label_accuracy"] == .412)
    s1 = json.loads((EXP / "01_scorer_selection" / "results" / "summary.json").read_text())
    check("Exp 01: AraBERT and mBERT both scored on 500 prompts", s1["config"]["n_prompts"] == 500
          and set(s1["results"]) >= {"arabert", "mbert"})


# ----------------------------------------------------------------- docs + code
def audit_docs():
    print("\n== F. Documentation and artefact")
    import subprocess
    rules = json.loads((PROJECT / "apcs" / "apcs" / "rules_default.json").read_text())
    check("shipped rules = documented APCS 1.0.0 (80 / 250, 0.5 / 0.3, τ 0.65)",
          rules["thresholds"] == {"T1": 80, "T2": 250, "r_mid": 0.5, "r_long": 0.3}
          and rules["fidelity_threshold_tau"] == 0.65)
    v1 = json.loads((E11 / "selectors" / "apcs_v1.json").read_text())
    check("exam APCS 1.0.0 selector identical to shipped rules", v1["thresholds"] == rules["thresholds"])
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"],
                       capture_output=True, text=True, cwd=PROJECT / "apcs")
    last = (r.stdout.strip().splitlines() or ["?"])[-1]
    check("apcs unit tests pass", r.returncode == 0, last)
    log = (EXP / "EXPERIMENT_LOG.md").read_text(encoding="utf-8")
    rows = [l for l in log.splitlines() if l.startswith("| ")]
    bad = [l[:40] for l in rows if len(l.split("|")) != 7]
    check("EXPERIMENT_LOG table rows well-formed", not bad, f"malformed: {bad}")
    for d in sorted(EXP.glob("[01]*_*")):
        check(f"{d.name}: README + FINDINGS present",
              (d / "README.md").exists() and (d / "results" / "FINDINGS.md").exists() or d.name.startswith("08"),
              "" if not d.name.startswith("08") else "08 records QC in results/qc_report.json + inspection_sample.md")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    v2, _ = audit_datasets()
    df = audit_exp10_raw(v2)
    audit_exp10_findings(df)
    audit_exp11()
    audit_supporting()
    audit_docs()
    c = Counter(s for s, _, _ in results)
    print(f"\nSUMMARY: {c['PASS']} pass, {c['WARN']} warn, {c['FAIL']} fail")
    sys.exit(1 if c["FAIL"] else 0)


if __name__ == "__main__":
    main()
