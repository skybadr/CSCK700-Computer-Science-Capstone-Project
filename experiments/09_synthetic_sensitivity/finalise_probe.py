"""Exp 09 prep — finalise the cross-category synthetic probe pool.

- merges the four probe part-files
- recomputes each prompt's ACTUAL cl100k length band (author estimates drifted:
  Arabic ~4.4 tokens/word)
- sentence-trims oversized summarisation passages to <=640 tokens (task stays
  valid); oversized qa prompts are kept intact but banded 'xlong' (questions may
  reference late facts, so trimming risks unanswerable items) — xlong rows are
  excluded from band-matched contrasts and enter length-covariate analyses only
- dedups against pilot + v2 + self
Output: probe_pool.json (quarantined: never merged into AraPromptBench v2)
"""

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent

PARTS = ["probe_instruction.json", "probe_qa.json",
         "probe_summarisation.json", "probe_short_addendum.json"]


def norm(s):
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[ً-ْـ]", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def band(n):
    if 30 <= n <= 90: return "short"
    if n <= 250: return "medium"
    if n <= 650: return "long"
    return "xlong"


def trim_summ(text, limit=640):
    """drop trailing sentences of the embedded passage until under limit."""
    while len(ENC.encode(text)) > limit:
        parts = re.split(r"(?<=[.؟!])\s+", text.rstrip("»«. "))
        if len(parts) < 3:
            break
        # remove the second-to-last sentence chunk (keep closing quote mark)
        tail = text.rstrip()
        m = list(re.finditer(r"[.؟!]", tail))
        if len(m) < 2:
            break
        text = tail[:m[-2].end()] + "»" if tail.endswith("»") else tail[:m[-2].end()]
    return text


def main():
    rows, seen = [], set()
    for src in [PROJECT / "AraPromptBench_dataset.json",
                PROJECT / "AraPromptBench_v2.json"]:
        for p in json.loads(src.read_text(encoding="utf-8"))["prompts"]:
            seen.add(norm(p["prompt"]))

    n_trimmed = 0
    for part in PARTS:
        for p in json.loads((ROOT / part).read_text(encoding="utf-8")):
            text = p["prompt"].strip()
            n = len(ENC.encode(text))
            if p["category"] == "summarisation" and n > 650:
                text = trim_summ(text)
                n = len(ENC.encode(text))
                n_trimmed += 1
            k = norm(text)
            if k in seen:
                print("  DUP dropped:", text[:40]); continue
            seen.add(k)
            rows.append(dict(category=p["category"], prompt=text,
                             token_count=n, length_band=band(n),
                             source="synthetic-claude-probe"))

    for i, r in enumerate(rows):
        r["id"] = f"probe-{r['category'][:4]}-{i+1:03d}"

    out = {"metadata": {
        "name": "AraPromptBench-synthetic-probe",
        "purpose": "Exp 09b: synthetic-vs-sourced comparison probe. "
                   "QUARANTINED — never merged into AraPromptBench v2, never "
                   "used for APCS calibration or the held-out test set.",
        "generator": "Claude Fable 5 (Anthropic), 2026-07",
        "note_xlong": "xlong (>650 cl100k tokens) rows enter "
                      "length-covariate analyses only, not band-matched "
                      "contrasts (no sourced counterparts in that range).",
    }, "prompts": rows}
    (ROOT / "probe_pool.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    cnt = Counter((r["category"], r["length_band"]) for r in rows)
    print(f"total: {len(rows)} | summ passages trimmed: {n_trimmed}")
    for c in ["instruction", "qa", "summarisation"]:
        print(f"  {c:14s}", {b: cnt.get((c, b), 0)
                             for b in ["short", "medium", "long", "xlong"]})
    print("saved: probe_pool.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
