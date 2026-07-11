"""Exp 08a — Source 850 corpus prompts for AraPromptBench v2 (final dataset).

Implements the SDR Table 2 composition from the six public sources, with:
  - per-prompt provenance (source, source_id, licence, variety)
  - deterministic sampling (seed 42) and template rotation
  - length-band quotas per category (short/medium/long by cl100k tokens of the
    FINAL prompt) to break the pilot's length<->category confound
  - exact-duplicate removal (normalised text) within v2 and against the pilot

The 150 synthetic creative prompts are built separately (Exp 08b) and merged
by assemble_v2.py. datasets 5.0 cannot run script-based loaders, so XL-Sum /
TyDi-QA / ARCD are read from the Hub's parquet conversion branches.

Output: results/sourced_prompts.json + printed band/category report.
"""

import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import tiktoken
from datasets import load_dataset

SEED = 42
ENC = tiktoken.get_encoding("cl100k_base")
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
PILOT = ROOT.parent.parent / "AraPromptBench_dataset.json"

# length bands measured on the final prompt text, cl100k tokens
BANDS = {"short": (30, 90), "medium": (91, 250), "long": (251, 650)}

SUMM_TEMPLATES = [
    "لخص النص التالي في ثلاث جمل كحد أقصى:\n\n{t}",
    "اقرأ المقال التالي ثم اكتب ملخصًا موجزًا له لا يتجاوز خمسين كلمة:\n\n{t}",
    "اكتب ملخصًا في فقرة واحدة يوضح الأفكار الرئيسية للنص الآتي:\n\n{t}",
    "لخص المقال التالي لقارئ مستعجل في جملتين فقط:\n\n{t}",
    "استخرج أهم النقاط من النص التالي وقدمها في ملخص قصير:\n\n{t}",
]
QA_TEMPLATES = [
    "اقرأ النص التالي ثم أجب عن السؤال:\n\n{c}\n\nالسؤال: {q}",
    "بناءً على الفقرة التالية، أجب بإيجاز:\n\n{c}\n\nالسؤال: {q}",
    "النص: {c}\n\nاعتمادًا على النص أعلاه، أجب عن السؤال التالي: {q}",
]


def ntok(s: str) -> int:
    return len(ENC.encode(s))


def band_of(n: int) -> str | None:
    for b, (lo, hi) in BANDS.items():
        if lo <= n <= hi:
            return b
    return None


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[ً-ْـ]", "", s)      # diacritics, tatweel
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def truncate_sentences(text: str, max_tokens: int) -> tuple[str, bool]:
    """Cut at sentence boundaries to fit a token budget."""
    if ntok(text) <= max_tokens:
        return text, False
    parts = re.split(r"(?<=[.؟!])\s+", text)
    out = ""
    for p in parts:
        cand = (out + " " + p).strip()
        if ntok(cand) > max_tokens:
            break
        out = cand
    return (out if out else text[:1000]), True


def arabic_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    return (sum("؀" <= c <= "ۿ" for c in letters) / len(letters)
            if letters else 0)


class Collector:
    def __init__(self):
        self.rows, self.seen = [], set()
        pilot = json.loads(PILOT.read_text(encoding="utf-8"))
        for p in pilot["prompts"]:
            self.seen.add(norm(p["prompt"]))

    def add(self, *, category, prompt, source, source_id, license_, variety,
            template_id=None, truncated=False, quota=None) -> bool:
        n = ntok(prompt)
        b = band_of(n)
        if b is None or arabic_ratio(prompt) < 0.8:
            return False
        if quota is not None and quota.get(b, 0) <= 0:
            return False
        key = norm(prompt)
        if not key or key in self.seen:
            return False
        self.seen.add(key)
        self.rows.append(dict(category=category, prompt=prompt, source=source,
                              source_id=str(source_id), license=license_,
                              variety=variety, length_band=b, token_count=n,
                              template_id=template_id, truncated=truncated))
        if quota is not None:
            quota[b] -= 1
        return True


def main() -> None:
    rng = random.Random(SEED)
    col = Collector()

    # ---------------- CIDAR: 250 instruction --------------------------------
    print("CIDAR ...", flush=True)
    dc = load_dataset("arbml/CIDAR", split="train")
    idx_cidar = list(range(len(dc)))
    rng.shuffle(idx_cidar)
    quota = {"short": 100, "medium": 100, "long": 50}
    for i in idx_cidar:
        if sum(quota.values()) == 0:
            break
        col.add(category="instruction", prompt=dc[i]["instruction"].strip(),
                source="CIDAR", source_id=dc[i]["index"],
                license_="CC BY-NC 4.0", variety="MSA", quota=quota)
    print(f"  instruction so far: {sum(r['category']=='instruction' for r in col.rows)}"
          f" (unfilled: {quota})")

    # ---------------- XL-Sum: 150 summarisation -----------------------------
    print("XL-Sum ...", flush=True)
    d = load_dataset("parquet", data_files="hf://datasets/csebuetnlp/xlsum@refs/convert/parquet/arabic/train/*.parquet",
                     split="train")
    idx = list(range(len(d)))
    rng.shuffle(idx)
    quota = {"short": 40, "medium": 55, "long": 55}
    for k, i in enumerate(idx):
        if sum(quota.values()) == 0:
            break
        text = d[i]["text"].strip()
        # fill bands: try natural fit first; truncate long articles for
        # whichever band still needs rows
        for target_b, (lo, hi) in BANDS.items():
            if quota.get(target_b, 0) <= 0:
                continue
            body, trunc = truncate_sentences(text, hi - 30)
            tmpl = SUMM_TEMPLATES[k % len(SUMM_TEMPLATES)]
            prompt = tmpl.format(t=body)
            if band_of(ntok(prompt)) == target_b:
                if col.add(category="summarisation", prompt=prompt,
                           source="XL-Sum", source_id=d[i]["id"],
                           license_="CC BY-NC-SA 4.0", variety="MSA",
                           template_id=k % len(SUMM_TEMPLATES),
                           truncated=trunc, quota=quota):
                    break
    print(f"  xlsum unfilled: {quota}")

    # ---------------- EASC: 100 summarisation -------------------------------
    print("EASC ...", flush=True)
    de = load_dataset("parquet", data_files="hf://datasets/arbml/easc@refs/convert/parquet/default/train/*.parquet",
                     split="train")
    idx_easc = list(range(len(de)))
    rng.shuffle(idx_easc)
    quota = {"short": 20, "medium": 45, "long": 35}
    for k, i in enumerate(idx_easc):
        if sum(quota.values()) == 0:
            break
        text = re.sub(r"^﻿", "", de[i]["article"].strip())
        for target_b, (lo, hi) in BANDS.items():
            if quota.get(target_b, 0) <= 0:
                continue
            body, trunc = truncate_sentences(text, hi - 30)
            tmpl = SUMM_TEMPLATES[(k + 2) % len(SUMM_TEMPLATES)]
            prompt = tmpl.format(t=body)
            if band_of(ntok(prompt)) == target_b:
                if col.add(category="summarisation", prompt=prompt,
                           source="EASC", source_id=i,
                           license_="research-use", variety="MSA",
                           template_id=(k + 2) % len(SUMM_TEMPLATES),
                           truncated=trunc, quota=quota):
                    break
    print(f"  easc unfilled: {quota}")

    # ---------------- TyDi-QA GoldP: 150 qa ---------------------------------
    print("TyDi-QA ...", flush=True)
    parts = []
    for split in ["train", "validation"]:
        dd = load_dataset("parquet", data_files=f"hf://datasets/google-research-datasets/tydiqa@refs/convert/parquet/secondary_task/{split}/*.parquet",
                          split="train")
        parts.append(dd.filter(lambda x: x["id"].startswith("arabic")))
    quota = {"short": 40, "medium": 55, "long": 55}
    pool_tydi = [(p, j) for p in parts for j in range(len(p))]
    rng.shuffle(pool_tydi)
    for k, (p, j) in enumerate(pool_tydi):
        if sum(quota.values()) == 0:
            break
        r = p[j]
        tmpl = QA_TEMPLATES[k % len(QA_TEMPLATES)]
        prompt = tmpl.format(c=r["context"].strip(), q=r["question"].strip())
        col.add(category="qa", prompt=prompt, source="TyDi-QA",
                source_id=r["id"], license_="Apache 2.0", variety="MSA",
                template_id=k % len(QA_TEMPLATES), quota=quota)
    print(f"  tydiqa unfilled: {quota}")

    # ---------------- ARCD: 100 qa ------------------------------------------
    print("ARCD ...", flush=True)
    parts = []
    for split in ["train", "validation"]:
        parts.append(load_dataset("parquet", data_files=f"hf://datasets/hsseinmz/arcd@refs/convert/parquet/plain_text/{split}/*.parquet",
                                  split="train"))
    quota = {"short": 25, "medium": 40, "long": 35}
    pool = [(p, j) for p in parts for j in range(len(p))]
    rng.shuffle(pool)
    for k, (p, j) in enumerate(pool):
        if sum(quota.values()) == 0:
            break
        r = p[j]
        tmpl = QA_TEMPLATES[(k + 1) % len(QA_TEMPLATES)]
        prompt = tmpl.format(c=r["context"].strip(), q=r["question"].strip())
        col.add(category="qa", prompt=prompt, source="ARCD",
                source_id=r["id"], license_="CC BY-SA 4.0", variety="MSA",
                template_id=(k + 1) % len(QA_TEMPLATES), quota=quota)
    # short-band backfill: truncate context at sentence boundaries, but only
    # keep the row if the gold answer span survives the cut (stays answerable)
    if quota.get("short", 0) > 0:
        for k, (p, j) in enumerate(pool):
            if quota["short"] <= 0:
                break
            r = p[j]
            answer = r["answers"]["text"][0] if r["answers"]["text"] else ""
            if not answer:
                continue
            body, trunc = truncate_sentences(r["context"].strip(), 55)
            if answer not in body:
                continue
            tmpl = QA_TEMPLATES[(k + 1) % len(QA_TEMPLATES)]
            prompt = tmpl.format(c=body, q=r["question"].strip())
            col.add(category="qa", prompt=prompt, source="ARCD",
                    source_id=r["id"], license_="CC BY-SA 4.0", variety="MSA",
                    template_id=(k + 1) % len(QA_TEMPLATES), truncated=True,
                    quota=quota)
    print(f"  arcd unfilled: {quota}")

    # ---------------- Aya: 100 mixed (80 MSA + 20 dialect) ------------------
    print("Aya ...", flush=True)
    d = load_dataset("parquet", data_files="hf://datasets/CohereForAI/aya_dataset@refs/convert/parquet/default/train/*.parquet",
                     split="train")
    msa = [i for i, c in enumerate(d["language_code"]) if c == "arb"]
    dia = [i for i, c in enumerate(d["language_code"]) if c in
           ("ary", "arz", "ars", "apc", "acq")]
    rng.shuffle(msa)
    rng.shuffle(dia)
    for pool_idx, n_target, variety_of in [
            (msa, 80, lambda r: "MSA"),
            (dia, 20, lambda r: f"dialect-{r['language_code']}")]:
        got = 0
        for i in pool_idx:
            if got >= n_target:
                break
            r = d[i]
            if col.add(category="instruction", prompt=r["inputs"].strip(),
                       source="Aya", source_id=r.get("user_id") or i,
                       license_="Apache 2.0", variety=variety_of(r)):
                got += 1
        print(f"  aya {'dialect' if pool_idx is dia else 'msa'}: {got}/{n_target}")

    # ---------------- backfill to source totals (band relaxed) --------------
    TARGETS = {"CIDAR": 250, "XL-Sum": 150, "EASC": 100, "TyDi-QA": 150,
               "ARCD": 100, "Aya": 100}
    import collections as _c
    have = _c.Counter(r["source"] for r in col.rows)
    for src_name, tgt in TARGETS.items():
        missing = tgt - have.get(src_name, 0)
        if missing <= 0:
            continue
        print(f"  backfill {src_name}: {missing} (band relaxed)", flush=True)
        if src_name == "CIDAR":
            for i in idx_cidar:
                if missing <= 0:
                    break
                if col.add(category="instruction",
                           prompt=dc[i]["instruction"].strip(),
                           source="CIDAR", source_id=dc[i]["index"],
                           license_="CC BY-NC 4.0", variety="MSA"):
                    missing -= 1
        elif src_name == "EASC":
            for k, i in enumerate(idx_easc):
                if missing <= 0:
                    break
                text = re.sub(r"^﻿", "", de[i]["article"].strip())
                body, trunc = truncate_sentences(text, 220)
                tmpl = SUMM_TEMPLATES[(k + 4) % len(SUMM_TEMPLATES)]
                if col.add(category="summarisation", prompt=tmpl.format(t=body),
                           source="EASC", source_id=i,
                           license_="research-use", variety="MSA",
                           template_id=(k + 4) % len(SUMM_TEMPLATES),
                           truncated=trunc):
                    missing -= 1

    # category top-up: if ARCD cannot reach its target (pool exhausted after
    # dedup/answerability), fill the qa category to 250 from TyDi-QA's deeper
    # pool — documented deviation from SDR Table 2 source counts
    have = _c.Counter(r["source"] for r in col.rows)
    qa_gap = 250 - sum(r["category"] == "qa" for r in col.rows)
    if qa_gap > 0:
        print(f"  topping up qa from TyDi-QA: {qa_gap}", flush=True)
        for k, (p, j) in enumerate(pool_tydi):
            if qa_gap <= 0:
                break
            r = p[j]
            tmpl = QA_TEMPLATES[k % len(QA_TEMPLATES)]
            prompt = tmpl.format(c=r["context"].strip(), q=r["question"].strip())
            if col.add(category="qa", prompt=prompt, source="TyDi-QA",
                       source_id=r["id"], license_="Apache 2.0", variety="MSA",
                       template_id=k % len(QA_TEMPLATES)):
                qa_gap -= 1

    # ---------------- report + save -----------------------------------------
    out = {"metadata": {
        "name": "AraPromptBench-v2-sourced", "seed": SEED,
        "bands_cl100k": {k: list(v) for k, v in BANDS.items()},
        "note": "corpus-sourced portion; creative synthetic added by assemble_v2.py",
    }, "prompts": col.rows}
    (RESULTS / "sourced_prompts.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    import collections
    print(f"\nTOTAL sourced: {len(col.rows)}")
    by = collections.Counter((r["category"], r["length_band"]) for r in col.rows)
    for cat in ["instruction", "summarisation", "qa"]:
        print(f"  {cat:14s} " + "  ".join(
            f"{b}:{by.get((cat, b), 0)}" for b in BANDS))
    src = collections.Counter(r["source"] for r in col.rows)
    print("  by source:", dict(src))
    var = collections.Counter(r["variety"] for r in col.rows)
    print("  by variety:", dict(var))
    print(f"\nSaved: {RESULTS / 'sourced_prompts.json'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
