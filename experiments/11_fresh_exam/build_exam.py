"""Exp 11a — Build the fresh exam set: ~400 never-seen prompts.

Purpose: a clean held-out test for the redesigned selectors (APCS-v2,
APCS-cost) and a replication test for APCS 1.0.0. The v2 test split was
already used once, so it cannot evaluate designs made after its results.

Same construction as AraPromptBench v2 (templates, sentence truncation,
length bands, Arabic-ratio gate) and the same source mix, scaled to 400:
  instruction 140 (CIDAR 100, Aya 32 MSA + 8 dialect)
  summarisation 100 (EASC 40, XL-Sum 60)
  qa 100 (ARCD 36, TyDi-QA 64)        -> gold answers kept for correctness
  creative 60 (new synthetic briefs, creative_exam.json)
Band quotas per category follow v2's band proportions.

Exclusions (stricter than v2): any record id used in v2; for QA any passage
used in v2 or already taken in this set (one question per passage); and
normalised-text duplicates of any pilot, v2, probe or exam prompt.
Seed 4242 (different from v2's 42, so sampling is independent).

Output: ../../AraPromptBench_exam.json, results/qc_report.json,
results/inspection_sample.md
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
sys.path.insert(0, str(ROOT.parent / "08_dataset_construction"))
from build_sourced import (BANDS, QA_TEMPLATES, SUMM_TEMPLATES,  # noqa: E402
                           arabic_ratio, band_of, norm, ntok,
                           truncate_sentences)

SEED = 4242
RESULTS = ROOT / "results"
OUT = PROJECT / "AraPromptBench_exam.json"
PQ = "hf://datasets/{}@refs/convert/parquet/{}"

# v2 band proportions, scaled to the exam's category sizes
QUOTA = {
    "instruction": {"short": 69, "medium": 50, "long": 21},
    "summarisation": {"short": 23, "medium": 41, "long": 36},
    "qa": {"short": 22, "medium": 38, "long": 40},
}
PREFIX = {"instruction": "inst", "summarisation": "summ", "qa": "qa",
          "creative": "creat"}


def ctx_key(text):
    return norm(text)[:200]


def shingles(text, k=12):
    w = norm(text).split()
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


# template wording is shared by design; only content text must not overlap
TEMPLATE_SH = set()
for _t in SUMM_TEMPLATES + QA_TEMPLATES:
    for _part in _t.replace("{t}", "|").replace("{c}", "|").replace("{q}", "|").split("|"):
        TEMPLATE_SH |= shingles(_part)


class Pool:
    def __init__(self):
        self.rows, self.seen, self.used_ctx = [], set(), set()
        for f in [PROJECT / "AraPromptBench_dataset.json",
                  PROJECT / "AraPromptBench_v2.json",
                  ROOT.parent / "09_synthetic_sensitivity" / "probe_pool.json"]:
            for p in json.loads(f.read_text(encoding="utf-8"))["prompts"]:
                self.seen.add(norm(p["prompt"]))
        v2 = json.loads((PROJECT / "AraPromptBench_v2.json").read_text(encoding="utf-8"))
        self.used_ids = {}
        self.used_sh = set()
        for p in v2["prompts"]:
            self.used_ids.setdefault(p["source"], set()).add(str(p["source_id"]))
            self.used_sh |= shingles(p["prompt"])
        self.used_sh -= TEMPLATE_SH

    def need(self, cat, band):
        have = sum(1 for r in self.rows if r["category"] == cat
                   and r["length_band"] == band)
        return QUOTA[cat][band] - have

    def add(self, *, category, prompt, source, source_id, license_, variety,
            band_check=True, context=None, gold=None, template_id=None,
            truncated=False):
        if str(source_id) in self.used_ids.get(source, set()):
            return False
        n = ntok(prompt)
        b = band_of(n)
        if b is None or arabic_ratio(prompt) < 0.8:
            return False
        if band_check and self.need(category, b) <= 0:
            return False
        if context is not None and ctx_key(context) in self.used_ctx:
            return False
        k = norm(prompt)
        if not k or k in self.seen:
            return False
        # partial overlap: any 12-word stretch of content text already used
        # by v2 or by an earlier exam prompt (catches re-cut passages)
        sh = shingles(prompt) - TEMPLATE_SH
        if sh & self.used_sh:
            return False
        self.seen.add(k)
        self.used_sh |= sh
        if context is not None:
            self.used_ctx.add(ctx_key(context))
        self.rows.append(dict(category=category, prompt=prompt, source=source,
                              source_id=str(source_id), license=license_,
                              variety=variety, length_band=b, token_count=n,
                              gold_answers=gold, template_id=template_id,
                              truncated=truncated))
        return True


def fill_summ(pool, rng, records, n_target, source, license_, text_of, id_of,
              tmpl_offset):
    got = 0
    for k, r in enumerate(records):
        if got >= n_target or all(pool.need("summarisation", b) <= 0
                                  for b in BANDS):
            break
        text = re.sub(r"^﻿", "", text_of(r).strip())
        for band, (lo, hi) in BANDS.items():
            if pool.need("summarisation", band) <= 0:
                continue
            body, trunc = truncate_sentences(text, hi - 30)
            t = (k + tmpl_offset) % len(SUMM_TEMPLATES)
            prompt = SUMM_TEMPLATES[t].format(t=body)
            if band_of(ntok(prompt)) == band and pool.add(
                    category="summarisation", prompt=prompt, source=source,
                    source_id=id_of(r, k), license_=license_, variety="MSA",
                    template_id=t, truncated=trunc):
                got += 1
                break
    return got


def fill_qa(pool, records, n_target, source, license_, id_of, tmpl_offset,
            short_truncate):
    got = 0
    for k, r in enumerate(records):
        if got >= n_target or all(pool.need("qa", b) <= 0 for b in BANDS):
            break
        gold = list(r["answers"]["text"]) if r["answers"]["text"] else []
        if not gold:
            continue
        t = (k + tmpl_offset) % len(QA_TEMPLATES)
        ctx = r["context"].strip()
        prompt = QA_TEMPLATES[t].format(c=ctx, q=r["question"].strip())
        ok = pool.add(category="qa", prompt=prompt, source=source,
                      source_id=id_of(r), license_=license_, variety="MSA",
                      context=ctx, gold=gold, template_id=t)
        if not ok and short_truncate and pool.need("qa", "short") > 0:
            # answer-preserving truncation into the short band
            body, _ = truncate_sentences(ctx, 55)
            if any(g in body for g in gold):
                ok = pool.add(category="qa", prompt=QA_TEMPLATES[t].format(
                    c=body, q=r["question"].strip()), source=source,
                    source_id=id_of(r), license_=license_, variety="MSA",
                    context=ctx, gold=gold, template_id=t, truncated=True)
        got += ok
    return got


def main():
    rng = random.Random(SEED)
    pool = Pool()

    # ---- instruction: Aya first (unbanded, as in v2), then CIDAR fills bands
    print("Aya ...", flush=True)
    aya = load_dataset("parquet", data_files=PQ.format(
        "CohereForAI/aya_dataset", "default/train/*.parquet"), split="train")
    msa = [i for i, c in enumerate(aya["language_code"]) if c == "arb"]
    dia = [i for i, c in enumerate(aya["language_code"])
           if c in ("ary", "arz", "ars")]
    rng.shuffle(msa)
    rng.shuffle(dia)
    for idx, n_target, var in [(msa, 32, lambda r: "MSA"),
                               (dia, 8, lambda r: f"dialect-{r['language_code']}")]:
        got = 0
        for i in idx:
            if got >= n_target:
                break
            r = aya[i]
            got += pool.add(category="instruction", prompt=r["inputs"].strip(),
                            source="Aya", source_id=f"row{i}",
                            license_="Apache 2.0", variety=var(r))
    print("CIDAR ...", flush=True)
    cid = load_dataset("arbml/CIDAR", split="train")
    order = list(range(len(cid)))
    rng.shuffle(order)
    for i in order:
        if all(pool.need("instruction", b) <= 0 for b in BANDS):
            break
        pool.add(category="instruction", prompt=cid[i]["instruction"].strip(),
                 source="CIDAR", source_id=cid[i]["index"],
                 license_="CC BY-NC 4.0", variety="MSA")
    # long instructions are rare in CIDAR: take long ones from the unused Aya
    # MSA pool first (keeps the band mix), then relax the band for any rest
    for i in msa[200:]:
        if pool.need("instruction", "long") <= 0:
            break
        r = aya[i]
        pool.add(category="instruction", prompt=r["inputs"].strip(),
                 source="Aya", source_id=f"row{i}", license_="Apache 2.0",
                 variety="MSA")
    short = sum(max(0, pool.need("instruction", b)) for b in BANDS)
    if short:
        print(f"  instruction band-relaxed backfill: {short}")
        for i in order:
            if short <= 0:
                break
            if pool.add(category="instruction",
                        prompt=cid[i]["instruction"].strip(), source="CIDAR",
                        source_id=cid[i]["index"], license_="CC BY-NC 4.0",
                        variety="MSA", band_check=False):
                short -= 1

    # ---- summarisation: EASC 40, then XL-Sum fills the bands
    print("EASC / XL-Sum ...", flush=True)
    easc = load_dataset("parquet", data_files=PQ.format(
        "arbml/easc", "default/train/*.parquet"), split="train")
    eo = list(range(len(easc)))
    rng.shuffle(eo)
    fill_summ(pool, rng, [easc[i] | {"_i": i} for i in eo], 40, "EASC",
              "research-use", lambda r: r["article"], lambda r, k: r["_i"], 1)
    xl = load_dataset("parquet", data_files=PQ.format(
        "csebuetnlp/xlsum", "arabic/*/*.parquet"), split="train")
    xo = list(range(len(xl)))
    rng.shuffle(xo)
    fill_summ(pool, rng, (xl[i] for i in xo), 100, "XL-Sum", "CC BY-NC-SA 4.0",
              lambda r: r["text"], lambda r, k: r["id"], 3)

    # ---- qa: ARCD 36, then TyDi-QA fills the bands (gold answers kept)
    print("ARCD / TyDi-QA ...", flush=True)
    arcd = [r for s in ["train", "validation"] for r in load_dataset(
        "parquet", data_files=PQ.format("hsseinmz/arcd", f"plain_text/{s}/*.parquet"),
        split="train")]
    tydi = [r for s in ["train", "validation"] for r in load_dataset(
        "parquet", data_files=PQ.format("google-research-datasets/tydiqa",
                                        f"secondary_task/{s}/*.parquet"),
        split="train") if r["id"].startswith("arabic")]
    # block every passage v2 already used, whatever question it carried
    for src, recs in [("ARCD", arcd), ("TyDi-QA", tydi)]:
        ids = pool.used_ids.get(src, set())
        pool.used_ctx |= {ctx_key(r["context"]) for r in recs
                          if str(r["id"]) in ids}
    print(f"  blocked v2 passages: {len(pool.used_ctx)}")
    rng.shuffle(arcd)
    fill_qa(pool, arcd, 36, "ARCD", "CC BY-SA 4.0", lambda r: r["id"], 1, True)
    rng.shuffle(tydi)
    fill_qa(pool, tydi, 100, "TyDi-QA", "Apache 2.0", lambda r: r["id"], 0, False)

    # ---- creative: new synthetic briefs (band = measured, not enforced)
    creative = json.loads((ROOT / "creative_exam.json").read_text(encoding="utf-8"))
    for i, c in enumerate(creative):
        text = c["prompt"].strip()
        k = norm(text)
        if k in pool.seen:
            print("  creative duplicate skipped:", i)
            continue
        pool.seen.add(k)
        n = ntok(text)
        pool.rows.append(dict(category="creative", prompt=text,
                              source="synthetic-claude", source_id=f"x{i+1:03d}",
                              license="CC BY 4.0", variety="MSA",
                              length_band=band_of(n) or "long", token_count=n,
                              gold_answers=None, template_id=None,
                              truncated=False))

    # ---- ids, save, QC
    final = []
    for cat in ["instruction", "summarisation", "qa", "creative"]:
        rows = [r for r in pool.rows if r["category"] == cat]
        rng.shuffle(rows)
        for j, r in enumerate(rows):
            r["id"] = f"exam-{PREFIX[cat]}-{j+1:03d}"
            r["split"] = "exam"
        final.extend(sorted(rows, key=lambda x: x["id"]))
    OUT.write_text(json.dumps({"metadata": {
        "name": "AraPromptBench-exam", "version": "1.0.0", "seed": SEED,
        "purpose": "Fresh held-out test for Exp 11 (APCS-v2, APCS-cost, "
                   "APCS 1.0.0 replication). Never used for design.",
        "construction": "same as AraPromptBench v2; excludes all v2 record "
                        "ids, all v2 QA passages, one question per passage, "
                        "and text duplicates of pilot/v2/probe",
        "total_prompts": len(final)}, "prompts": final},
        ensure_ascii=False, indent=1), encoding="utf-8")

    RESULTS.mkdir(exist_ok=True)
    qc = {"total": len(final),
          "by_category": dict(Counter(r["category"] for r in final)),
          "by_source": dict(Counter(r["source"] for r in final)),
          "by_variety": dict(Counter(r["variety"] for r in final)),
          "band_by_category": {c: dict(Counter(r["length_band"] for r in final
                                               if r["category"] == c))
                               for c in PREFIX},
          "qa_with_gold": sum(1 for r in final if r["gold_answers"]),
          "qa_passages_distinct": len({ctx_key(r["prompt"]) for r in final
                                       if r["category"] == "qa"})}
    (RESULTS / "qc_report.json").write_text(json.dumps(qc, indent=2,
                                                       ensure_ascii=False))
    rng2 = random.Random(SEED + 1)
    lines = ["# Fresh exam set — 5% inspection sample (author review)\n",
             "Check each for fluency, task clarity and category fit; list any "
             "id to replace.\n"]
    for cat, n in [("instruction", 7), ("summarisation", 5), ("qa", 5),
                   ("creative", 3)]:
        lines.append(f"\n## {cat}\n")
        for r in rng2.sample([r for r in final if r["category"] == cat], n):
            gold = f"  \n*gold answer:* {r['gold_answers'][0]}" if r["gold_answers"] else ""
            lines.append(f"**{r['id']}** ({r['source']}, {r['length_band']})\n\n"
                         f"> {r['prompt'][:700]}{gold}\n")
    (RESULTS / "inspection_sample.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(qc, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
